from __future__ import annotations

import io
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.exchange_sync import ExchangeReconciler
from trading_agent.core.logging import get_logger
from trading_agent.core.reporting import report_markdown
from trading_agent.core.storage import Store
from trading_agent.exchange import BinanceSpotAdapter
from trading_agent.graph import SupervisorRuntime
from trading_agent.repl.chat import handle_chat
from trading_agent.repl.events import EventBus
from trading_agent.repl.lifecycle import AgentLifecycleManager
from trading_agent.repl.renderer import AgentRenderer
from trading_agent.utils.binance_skills import BinanceSkillRegistry
from trading_agent.utils.logtail import follow, read_last_lines
from trading_agent.core.pnl import unrealized_pnl
from trading_agent.utils.market_data import cached_current_prices, current_prices
from trading_agent.utils.web_search import run_web_news_search, run_web_search

LOGGER = get_logger("repl")

_HISTORY_PATH = Path.home() / ".trading_agent_history"

def build_console() -> Console:
    """Build a console that cannot crash on streamed LLM/exchange Unicode.

    - Piped/redirected stdout: rich's legacy Win32 styled writer cannot query
      a pipe's screen buffer, and the locale cp1252 encoding chokes on token
      names and LLM glyphs - write UTF-8 with replacement.
    - Interactive legacy Windows consoles (cp1252 etc.): keep the console
      encoding, but replace unencodable characters instead of raising
      UnicodeEncodeError (which previously killed the UI pump thread).
    """
    if not sys.stdout.isatty():
        stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        return Console(file=stream, legacy_windows=False, force_terminal=False, no_color=True)
    encoding = (sys.stdout.encoding or "utf-8").lower().replace("-", "")
    if encoding in {"utf8", "cp65001"}:
        return Console()
    stream = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding=sys.stdout.encoding,
        errors="replace",
        line_buffering=True,
    )
    return Console(file=stream)


_COMMANDS = {
    "/chat <message>": "Talk to the supervisor team (price queries, open/close orders, anything). Plain text without a leading / also chats.",
    "/run [--symbols S ...] [--interval N]": "Start the agent loop in the background.",
    "/once [--symbols S ...]": "Run exactly one supervised cycle.",
    "/pause": "Pause the loop at the next safe boundary (NemoClaw lifecycle).",
    "/resume": "Resume a paused loop.",
    "/stop": "Gracefully stop the loop (finishes the current cycle).",
    "/status": "Show lifecycle state, mode, model, positions, and kill switch.",
    "/balance": "Show exchange account balances.",
    "/skills [list | run <name> <cmd> <json>]": "Inspect or invoke approved read-only skills.",
    "/search <query>": "Live web search (free DuckDuckGo).",
    "/news <query>": "Live news search (last 24h).",
    "/orders [N] [--sync]": "Show recent orders; --sync refreshes live exchange status first (auto in testnet/live).",
    "/logs [N | follow]": "Show the last N log lines, or stream the log live (Ctrl-C to stop).",
    "/kill on|off": "Toggle the emergency kill switch.",
    "/report": "Render the markdown operations report.",
    "/help": "Show this command list.",
    "/exit": "Quit the REPL (Ctrl-D also works).",
}


class TradingAgentREPL:
    """Interactive operator console: streaming agent activity + slash commands."""

    def __init__(
        self,
        config: AppConfig,
        settings: Settings,
        *,
        symbols: list[str] | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.symbols = symbols or list(config.risk.allowed_symbols)
        self.console = build_console()
        self.bus = EventBus()
        self.lifecycle = AgentLifecycleManager(self.bus)
        self.renderer = AgentRenderer(self.console)
        self.registry = BinanceSkillRegistry()
        self._agent_thread: threading.Thread | None = None
        self._ui_stop = threading.Event()
        self._cycle = 0
        self._last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        pump = threading.Thread(target=self._ui_pump, name="repl-ui-pump", daemon=True)
        pump.start()
        self._print_banner()
        try:
            if sys.stdin.isatty():
                self._interactive_loop()
            else:
                self._piped_loop()
        finally:
            self.lifecycle.stop()
            self._ui_stop.set()
            if self._agent_thread is not None:
                self._agent_thread.join(timeout=30)
            pump.join(timeout=5)  # let the pump finish printing before shutdown
            self.renderer.flush_tokens()
        self.console.print("[dim]bye[/dim]")
        return 0

    def _interactive_loop(self) -> None:
        session: PromptSession[str] = PromptSession(history=FileHistory(str(_HISTORY_PATH)))
        with patch_stdout(raw=True):
            while True:
                try:
                    line = session.prompt("> ", bottom_toolbar=self._toolbar)
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                if not line.strip():
                    continue
                if not self._dispatch(line.strip()):
                    break

    def _piped_loop(self) -> None:
        """Scriptable mode: read commands from piped stdin, no terminal UI."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            # PowerShell pipes may prepend a BOM, decoded as U+FEFF or as
            # mojibake bytes depending on the active codepage.
            for bom in ("\ufeff", "\u00ef\u00bb\u00bf"):
                line = line.removeprefix(bom)
            line = line.strip()
            if not line:
                continue
            self.console.print(f"[dim]> {line}[/dim]")
            if not self._dispatch(line):
                break

    def _print_banner(self) -> None:
        self.renderer.render_header(self._status_payload())
        self._print_balances()
        self._print_open_positions()
        self.console.print(
            "[dim]Type /run to start the agent loop, /help for commands.[/dim]"
        )

    def _fetch_balances(self) -> list[dict[str, Any]] | None:
        if self.settings.binance_api_key is None or self.settings.binance_api_secret is None:
            return None
        try:
            credentials = BinanceSpotAdapter.credentials_from_env(settings=self.settings)
            adapter = BinanceSpotAdapter(base_url=self.settings.exchange_base_url(), settings=self.settings)
            return adapter.account_balances(credentials)
        except Exception as exc:
            LOGGER.warning("repl balance fetch failed: %s", exc)
            self.console.print(f"[yellow]balance fetch failed: {exc}[/yellow]")
            return None

    def _print_balances(self) -> None:
        balances = self._fetch_balances()
        if balances is None:
            self.console.print("[dim]balances: unavailable (no exchange credentials configured)[/dim]")
            return
        if not balances:
            self.console.print("[yellow]balances: account is empty[/yellow]")
            return
        # Surface the assets we trade first; testnet accounts hold dozens more.
        priority = {"USDT", "BTC", "ETH", "SOL", "BNB"}
        shown = [item for item in balances if item["asset"] in priority]
        others = [item for item in balances if item["asset"] not in priority]
        shown.extend(others[: max(0, 12 - len(shown))])
        line = "  ".join(
            f"{item['asset']}: {item['free']:g}" + (f" (locked {item['locked']:g})" if item["locked"] else "")
            for item in shown
        )
        hidden = len(balances) - len(shown)
        if hidden > 0:
            line += f"  (+{hidden} more)"
        self.console.print(f"[bold]balances[/bold]  {line}")

    def _print_open_positions(self) -> None:
        with Store(self.config.database_path) as store:
            positions = store.open_positions()
            prices = cached_current_prices(
                [o.symbol for o in positions], store, ttl=self.config.risk.mark_refresh_seconds
            )
        if not positions:
            self.console.print("[dim]open positions: none[/dim]")
            return
        unrealized_total = 0.0
        for order in positions:
            price = prices.get(order.symbol.upper())
            up, up_pct = unrealized_pnl(order, price)
            line = (
                f"[bold]open[/bold]  {order.symbol} {order.side.value} qty={order.quantity:g} "
                f"@ {order.price:g}  TP {order.take_profit_price:g} / SL {order.stop_loss_price:g} "
                f"since {order.opened_at[:19]}"
            )
            if up is not None:
                unrealized_total += up
                line += f"  live {price:g} uPnL {up:+.4f} ({up_pct:+.2f}%)"
            self.console.print(line)
        self.console.print(f"[bold]unrealized total[/bold] {unrealized_total:+.4f} USDT")

    def _toolbar(self) -> str:
        return (
            f" {self.lifecycle.state.upper()} | cycle {self._cycle} | "
            f"mode {self.settings.trading_agent_execution_mode} | "
            f"model {self.settings.model_provider}:{self.settings.resolved_model_name()}"
        )

    def _ui_pump(self) -> None:
        try:
            while not self._ui_stop.wait(timeout=0.2):
                for event in self.bus.drain():
                    self._render_safely(event)
            for event in self.bus.drain():
                self._render_safely(event)
        except Exception:  # stdout can close during interpreter shutdown
            LOGGER.exception("repl ui pump stopped")

    def _render_safely(self, event: Any) -> None:
        """One bad event must not kill the pump (and with it all live output)."""
        try:
            self.renderer.render_event(event)
        except Exception:
            LOGGER.exception("repl event render failed kind=%s", getattr(event, "kind", "?"))

    # ------------------------------------------------------------- dispatch

    def _dispatch(self, line: str) -> bool:
        if not line.startswith("/"):
            # Anything that isn't a slash command is a chat message to the team.
            self._chat(line)
            return True
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.console.print(f"[red]parse error: {exc}[/red]")
            return True
        command, args = parts[0].lower(), parts[1:]
        if command in {"/exit", "/quit"}:
            return False
        if command == "/chat":
            self._chat(" ".join(args) if args else "")
            return True
        handler = {
            "/run": self._cmd_run,
            "/once": self._cmd_once,
            "/pause": self._cmd_pause,
            "/resume": self._cmd_resume,
            "/stop": self._cmd_stop,
            "/status": self._cmd_status,
            "/balance": self._cmd_balance,
            "/skills": self._cmd_skills,
            "/search": self._cmd_search,
            "/news": self._cmd_news,
            "/orders": self._cmd_orders,
            "/logs": self._cmd_logs,
            "/kill": self._cmd_kill,
            "/report": self._cmd_report,
            "/help": self._cmd_help,
        }.get(command)
        if handler is None:
            self.console.print(f"[red]unknown command: {command}[/red] -- try /help")
            return True
        try:
            handler(args)
        except Exception as exc:
            LOGGER.exception("repl command failed command=%s", command)
            self.console.print(f"[bold red]error:[/bold red] {exc}")
        return True

    # ------------------------------------------------------------- commands

    def _cmd_run(self, args: list[str]) -> None:
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self.console.print("[yellow]agent loop already running -- /pause or /stop first[/yellow]")
            return
        symbols, interval = self._parse_run_args(args)
        self.lifecycle.starting()
        self._agent_thread = threading.Thread(
            target=self._agent_loop,
            args=(symbols, interval),
            name="repl-agent-loop",
            daemon=True,
        )
        self._agent_thread.start()

    def _cmd_once(self, args: list[str]) -> None:
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self.console.print("[yellow]agent loop already running[/yellow]")
            return
        symbols, _ = self._parse_run_args(args)
        self.lifecycle.starting()
        self._agent_thread = threading.Thread(
            target=self._agent_loop,
            args=(symbols, 0.0, 1),
            name="repl-agent-once",
            daemon=True,
        )
        self._agent_thread.start()
        # /once is a blocking, single-cycle command: wait so the streamed
        # output lands before the next prompt.
        self._agent_thread.join(timeout=600)

    def _cmd_pause(self, args: list[str]) -> None:
        self.lifecycle.pause()

    def _cmd_resume(self, args: list[str]) -> None:
        self.lifecycle.resume()

    def _cmd_stop(self, args: list[str]) -> None:
        self.lifecycle.stop()

    def _cmd_status(self, args: list[str]) -> None:
        self.renderer.render_header(self._status_payload())
        self._print_balances()
        self._print_open_positions()

    def _cmd_balance(self, args: list[str]) -> None:
        self._print_balances()
        self._print_open_positions()

    def _cmd_skills(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            for skill in self.registry.list_skills():
                commands = ", ".join(skill.commands) if skill.commands else "-"
                self.console.print(
                    f"[bold cyan]{skill.name}[/bold cyan] [{skill.category}] {commands}"
                )
            return
        if args[0] == "run":
            if len(args) < 4:
                self.console.print("usage: /skills run <name> <command> '<json>'")
                return
            name, command, params = args[1], args[2], args[3]
            if not self.lifecycle.check_skill_allowed(name, command):
                self.console.print(f"[bold red]policy: {name}/{command} is not allowlisted[/bold red]")
                return
            result = self.registry.run_read_only_cli(name, command, params)
            style = "green" if result.returncode == 0 else "red"
            self.console.print(f"[{style}]returncode={result.returncode}[/{style}]")
            if result.stdout:
                self.console.print(result.stdout[:4000])
            if result.stderr:
                self.console.print(f"[red]{result.stderr[:1000]}[/red]")
            return
        self.console.print("usage: /skills [list | run <name> <command> '<json>']")

    def _cmd_search(self, args: list[str]) -> None:
        if not args:
            self.console.print("usage: /search <query>")
            return
        for row in run_web_search(" ".join(args), max_results=8):
            self.console.print(f"[bold]{row['title']}[/bold]\n  [dim]{row['url']}[/dim]\n  {row['snippet'][:200]}")

    def _cmd_news(self, args: list[str]) -> None:
        if not args:
            self.console.print("usage: /news <query>")
            return
        for row in run_web_news_search(" ".join(args), timelimit="d", max_results=8):
            self.console.print(
                f"[bold]{row['title']}[/bold] [dim]({row['source']} {row['date']})[/dim]\n"
                f"  [dim]{row['url']}[/dim]"
            )

    def _chat(self, message: str) -> None:
        message = message.strip()
        if not message:
            self.console.print("usage: /chat <message>  (or just type without a leading /)")
            return
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self.console.print(
                "[yellow]agent loop is running; chat will wait for the current cycle to finish[/yellow]"
            )
        try:
            with Store(self.config.database_path) as store:
                runtime = SupervisorRuntime(self.config, store, settings=self.settings)
                runtime.event_callback = self.bus.emit_now
                handle_chat(
                    runtime,
                    self.renderer,
                    self.console,
                    message,
                    confirm=self._confirm,
                )
        except Exception as exc:
            LOGGER.exception("chat failed")
            self.console.print(f"[bold red]chat error:[/bold red] {exc}")

    def _confirm(self, question: str) -> bool:
        if not sys.stdin.isatty():
            self.console.print(f"[yellow]{question} -> declined (non-interactive session)[/yellow]")
            return False
        try:
            answer = input(question)
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    def _cmd_orders(self, args: list[str]) -> None:
        flags = [a for a in args if a.startswith("--")]
        numbers = [a for a in args if not a.startswith("--")]
        limit = int(numbers[0]) if numbers else 20
        want_sync = "--sync" in flags or self.settings.trading_agent_execution_mode in {"testnet", "live"}
        with Store(self.config.database_path) as store:
            if want_sync:
                reconciler = ExchangeReconciler(store, self.settings)
                if reconciler.available:
                    updated = reconciler.reconcile()
                    if updated:
                        self.console.print(f"[dim]synced {len(updated)} order(s) from exchange[/dim]")
            orders = store.all_orders(limit)
            open_orders = store.open_positions()
            prices = cached_current_prices(
                [o.symbol for o in open_orders], store, ttl=self.config.risk.mark_refresh_seconds
            )
            marks: dict[str, tuple[float | None, float | None, float | None]] = {}
            unrealized_total = 0.0
            for o in open_orders:
                price = prices.get(o.symbol.upper())
                up, up_pct = unrealized_pnl(o, price)
                marks[o.id] = (price, up, up_pct)
                if up is not None:
                    unrealized_total += up
        if not orders:
            self.console.print("[dim]no orders yet[/dim]")
            return
        for order in orders:
            line = (
                f"{order.get('opened_at', '')[:19]} {order.get('mode')} {order.get('symbol')} "
                f"{order.get('side')} @ {order.get('price')} qty={order.get('quantity')} "
                f"status={order.get('status')}"
            )
            if order.get("exchange_status"):
                line += f" exchange={order.get('exchange_status')}"
            if order.get("avg_fill_price"):
                line += f" fill@{order.get('avg_fill_price')}"
            if order.get("cumulative_quote_qty"):
                line += f" spent={order.get('cumulative_quote_qty')}"
            line += f" realized={order.get('realized_pnl')}"
            if order.get("pnl_estimated"):
                line += " (est.)"
            price, up, up_pct = marks.get(order.get("id"), (None, None, None))
            if up is not None:
                line += f" | live {price:g} unrealized={up:+.4f} ({up_pct:+.2f}%)"
            self.console.print(line)
        if open_orders:
            self.console.print(
                f"[bold]unrealized total[/bold] {unrealized_total:+.4f} USDT "
                f"across {len(open_orders)} open position(s)"
            )

    def _cmd_logs(self, args: list[str]) -> None:
        log_path = Path(self.config.home) / "logs" / "trading_agent.log"
        if not log_path.exists():
            self.console.print(f"[dim]no log file yet at {log_path}[/dim]")
            return
        if args and args[0] in {"follow", "-f", "--follow"}:
            self.console.print(f"[dim]following {log_path} -- Ctrl-C to stop[/dim]")
            try:
                for line in follow(log_path):
                    self.console.print(line, markup=False, highlight=False)
            except KeyboardInterrupt:
                self.console.print("[dim]stopped following[/dim]")
            return
        count = int(args[0]) if args else 50
        for line in read_last_lines(log_path, count):
            self.console.print(line, markup=False, highlight=False)

    def _cmd_kill(self, args: list[str]) -> None:
        if not args or args[0] not in {"on", "off"}:
            self.console.print("usage: /kill on|off")
            return
        enabled = args[0] == "on"
        with Store(self.config.database_path) as store:
            store.set_setting("kill_switch", enabled)
            store.log_event("kill_switch_changed", {"enabled": enabled, "source": "repl"})
        style = "bold red" if enabled else "bold green"
        self.console.print(f"[{style}]kill switch {'ON' if enabled else 'OFF'}[/{style}]")

    def _cmd_report(self, args: list[str]) -> None:
        with Store(self.config.database_path) as store:
            prices = current_prices([order.symbol for order in store.open_positions()])
            self.console.print(report_markdown(store, self.config, prices=prices))

    def _cmd_help(self, args: list[str]) -> None:
        self.renderer.render_help(_COMMANDS)

    # ----------------------------------------------------------- agent loop

    def _agent_loop(self, symbols: list[str], interval: float, max_cycles: int | None = None) -> None:
        completed = 0
        try:
            with Store(self.config.database_path) as store:
                checkpoint = store.get_setting("agent_checkpoint", None) or {}
                self._cycle = int(checkpoint.get("last_cycle", -1)) + 1
                runtime = SupervisorRuntime(self.config, store, settings=self.settings)
                runtime.event_callback = self.bus.emit_now
                self.lifecycle.running()
                while self.lifecycle.wait_or_die():
                    if store.get_setting("kill_switch", False):
                        self.bus.emit_now(
                            "info", "lifecycle", {"message": "kill switch is ON -- cycle will reject intents"}
                        )
                    try:
                        result = runtime.run_once(cycle=self._cycle, symbols=symbols)
                        self._last_result = {
                            "cycle": result.cycle,
                            "evidence": result.evidence_count,
                            "intents": result.intent_count,
                            "approved": result.approved_count,
                            "rejected": result.rejected_count,
                            "submitted": result.submitted_trades,
                            "errors": result.error_count,
                        }
                        self.bus.emit_now("info", "supervisor", {"message": f"cycle {result.cycle} done: {self._last_result}"})
                    except Exception as exc:
                        LOGGER.exception("repl agent cycle failed cycle=%s", self._cycle)
                        self.bus.emit_now("error", "supervisor", {"message": f"cycle {self._cycle} failed: {exc}"})
                    self._cycle += 1
                    completed += 1
                    if max_cycles is not None and completed >= max_cycles:
                        break
                    if interval > 0:

                        def _monitor_tick() -> None:
                            if not store.get_setting("kill_switch", False):
                                try:
                                    runtime.monitor_open_positions()
                                except Exception:
                                    LOGGER.exception("repl bracket monitor tick failed")

                        if not self.lifecycle.sleep_or_die(
                            interval,
                            on_tick=_monitor_tick,
                            tick_seconds=max(15.0, float(self.config.risk.bracket_monitor_seconds)),
                        ):
                            break
        except Exception as exc:
            LOGGER.exception("repl agent loop crashed")
            self.bus.emit_now("error", "lifecycle", {"message": f"agent loop crashed: {exc}"})
        finally:
            self.lifecycle.stopped()

    # -------------------------------------------------------------- helpers

    def _parse_run_args(self, args: list[str]) -> tuple[list[str], float]:
        symbols = list(self.symbols)
        interval = float(self.config.decision_interval_minutes * 60)
        index = 0
        while index < len(args):
            if args[index] == "--symbols":
                index += 1
                collected: list[str] = []
                while index < len(args) and not args[index].startswith("--"):
                    collected.append(args[index].upper())
                    index += 1
                if collected:
                    symbols = collected
            elif args[index] == "--interval":
                index += 1
                if index < len(args):
                    interval = float(args[index])
                    index += 1
            else:
                index += 1
        return symbols, interval

    def _status_payload(self) -> dict[str, Any]:
        with Store(self.config.database_path) as store:
            kill = bool(store.get_setting("kill_switch", False))
            summary = store.summary()
        open_positions = summary.get("open_positions", summary.get("open_orders", "?"))
        return {
            "state": self.lifecycle.state.upper(),
            "cycle": self._cycle,
            "mode": self.settings.trading_agent_execution_mode,
            "live data": "ON" if self.settings.live_data else "OFF (simulated)",
            "llm supervisor": "ON" if self.settings.enable_llm_supervisor else "OFF",
            "model": f"{self.settings.model_provider}:{self.settings.resolved_model_name()}",
            "symbols": " ".join(self.symbols),
            "kill switch": "ON" if kill else "OFF",
            "open positions": open_positions,
            "last cycle": str(self._last_result or "-"),
        }


