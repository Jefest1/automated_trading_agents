from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_agent.agents.introductions import agent_introduction_payload
from trading_agent.backtest import Backtester
from trading_agent.decision_replay import candles_from_klines, replay_recorded_decisions
from trading_agent.core.config import DEFAULT_HOME, Settings, ensure_config, load_settings, save_config
from trading_agent.utils.llm_factory import require_model_api_key
from trading_agent.core.exchange_sync import ExchangeReconciler
from trading_agent.core.logging import configure_logging, get_logger, shutdown_logging
from trading_agent.core.models import new_id
from trading_agent.utils.logtail import follow, read_last_lines
from trading_agent.core.pnl import unrealized_pnl
from trading_agent.utils.market_data import cached_current_prices, current_prices
from trading_agent.core.reporting import report_json, report_markdown
from trading_agent.core.storage import Store
from trading_agent.exchange import BinanceSpotAdapter
from trading_agent.graph import SupervisorRuntime
from trading_agent.repl.app import TradingAgentREPL
from trading_agent.utils.binance_skills import BinanceSkillRegistry
from trading_agent.utils.mcp_tools import MCPToolLoader, load_mcp_config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    finally:
        shutdown_logging()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-agent")
    parser.add_argument(
        "--home",
        default=None,
        help="State directory for config and SQLite database. Defaults to TRADING_AGENT_HOME or .trading_agent.",
    )
    parser.add_argument("--env-file", default=None, help="Optional .env-style file parsed by trading_agent.core.config.")
    parser.add_argument("--log-level", default=None, help="Override TRADING_AGENT_LOG_LEVEL for this command.")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="Create default config and database.")
    init.set_defaults(handler=cmd_init)

    status = sub.add_parser("status", help="Show runtime status.")
    status.set_defaults(handler=cmd_status)

    repl = sub.add_parser("repl", help="Launch the interactive REPL (streaming agent activity + slash commands).")
    repl.add_argument("--symbols", nargs="*", default=None)
    repl.set_defaults(handler=cmd_repl)

    exchange = sub.add_parser("exchange", help="Exchange integration commands.")
    exchange_sub = exchange.add_subparsers(dest="exchange_command")
    ticker = exchange_sub.add_parser("ticker", help="Fetch a public Spot/Testnet ticker price.")
    ticker.add_argument("--symbol", required=True)
    ticker.set_defaults(handler=cmd_exchange_ticker)
    testnet_order = exchange_sub.add_parser(
        "testnet-limit-order",
        help="Validate or submit a hard-gated Binance Spot Testnet LIMIT order.",
    )
    testnet_order.add_argument("--symbol", required=True)
    testnet_order.add_argument("--side", required=True, choices=["BUY", "SELL"])
    testnet_order.add_argument("--quantity", required=True, help="Base asset quantity, e.g. 0.001")
    testnet_order.add_argument("--price", required=True, help="Limit price, e.g. 90000.00")
    testnet_order.add_argument("--client-order-id", default=None)
    testnet_order.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit to Spot Testnet. Without this, only /api/v3/order/test is called.",
    )
    testnet_order.set_defaults(handler=cmd_exchange_testnet_limit_order)

    agent = sub.add_parser("agent", help="LangGraph/Deep Agents supervised runtime commands.")
    agent_sub = agent.add_subparsers(dest="agent_command")
    agent_intro = agent_sub.add_parser("introduce", help="Show agent introductions and ask for starting tokens.")
    agent_intro.set_defaults(handler=cmd_agent_introduce)

    agent_once = agent_sub.add_parser("once", help="Run one supervised agent cycle.")
    agent_once.add_argument("--cycle", type=int, default=0)
    agent_once.add_argument("--symbols", nargs="*", default=None)
    agent_once.add_argument("--thread-id", default=None)
    agent_once.set_defaults(handler=cmd_agent_once)

    agent_run = agent_sub.add_parser("run", help="Run the supervised agent foreground loop.")
    _add_agent_loop_args(agent_run)
    agent_run.set_defaults(handler=cmd_agent_run)

    agent_daemon = agent_sub.add_parser("daemon", help="Run the supervised loop as a WSL/Linux foreground daemon.")
    _add_agent_loop_args(agent_daemon)
    agent_daemon.set_defaults(handler=cmd_agent_daemon)

    agent_status = agent_sub.add_parser("status", help="Show supervised runtime status.")
    agent_status.set_defaults(handler=cmd_agent_status)

    agent_stop = agent_sub.add_parser("stop", help="Request graceful supervised runtime shutdown.")
    agent_stop.set_defaults(handler=cmd_agent_stop)

    agent_skills = agent_sub.add_parser("skills", help="Inspect and run installed agent skills.")
    agent_skills_sub = agent_skills.add_subparsers(dest="agent_skills_command")
    agent_skills_list = agent_skills_sub.add_parser("list", help="List installed Binance Skills Hub skills.")
    agent_skills_list.set_defaults(handler=cmd_agent_skills_list)
    agent_skills_show = agent_skills_sub.add_parser("show", help="Show an installed skill's SKILL.md.")
    agent_skills_show.add_argument("name")
    agent_skills_show.set_defaults(handler=cmd_agent_skills_show)
    agent_skills_commands = agent_skills_sub.add_parser(
        "commands",
        help="Show approved read-only Binance skill CLI commands.",
    )
    agent_skills_commands.add_argument("name", nargs="?")
    agent_skills_commands.set_defaults(handler=cmd_agent_skills_commands)
    agent_skills_run = agent_skills_sub.add_parser(
        "run",
        help="Run an approved read-only Binance Web3 skill CLI command.",
    )
    agent_skills_run.add_argument("name")
    agent_skills_run.add_argument("skill_command")
    agent_skills_run.add_argument("params_json")
    agent_skills_run.set_defaults(handler=cmd_agent_skills_run)

    signals = sub.add_parser("signals", help="Show recent evidence records.")
    signals.add_argument("--limit", type=int, default=20)
    signals.set_defaults(handler=cmd_signals)

    orders = sub.add_parser("orders", help="Show recent orders.")
    orders.add_argument("--limit", type=int, default=50)
    orders.add_argument(
        "--sync",
        action="store_true",
        help="Refresh open testnet/live orders from the exchange before listing (live status, fills, PnL).",
    )
    orders.set_defaults(handler=cmd_orders)

    logs = sub.add_parser("logs", help="Show or follow the agent log file.")
    logs.add_argument("--lines", type=int, default=50, help="Number of trailing lines to print.")
    logs.add_argument("--follow", "-f", action="store_true", help="Stream new lines live (Ctrl-C to stop).")
    logs.set_defaults(handler=cmd_logs)

    risk = sub.add_parser("risk", help="Risk commands.")
    risk_sub = risk.add_subparsers(dest="risk_command")
    risk_config = risk_sub.add_parser("config", help="Show risk config.")
    risk_config.set_defaults(handler=cmd_risk_config)

    kill = sub.add_parser("kill-switch", help="Enable or disable the kill switch.")
    kill.add_argument("state", choices=["on", "off"])
    kill.set_defaults(handler=cmd_kill_switch)

    report = sub.add_parser("report", help="Render promotion/report output.")
    report.add_argument("--format", choices=["json", "markdown"], default="json")
    report.add_argument("--output")
    report.set_defaults(handler=cmd_report)

    backtest = sub.add_parser(
        "backtest",
        help="Replay historical klines through the deterministic strategy + risk pipeline.",
    )
    backtest.add_argument("--symbols", nargs="*", default=None)
    backtest.add_argument("--interval", default="1h", help="Kline interval (e.g. 15m, 1h, 4h).")
    backtest.add_argument("--limit", type=int, default=500, help="Number of candles (max 1000).")
    backtest.set_defaults(handler=cmd_backtest)

    replay = sub.add_parser(
        "backtest-decisions",
        help="Replay the supervisor's RECORDED decisions against the price path that followed each.",
    )
    replay.add_argument("--interval", default="1h", help="Kline interval for the forward window.")
    replay.add_argument(
        "--window", type=int, default=48, help="Candles to score after each decision (forward horizon)."
    )
    replay.add_argument(
        "--entry-ttl", type=int, default=3, help="Candles a limit entry stays live before expiring."
    )
    replay.set_defaults(handler=cmd_backtest_decisions)

    mcp_check = sub.add_parser(
        "mcp-check",
        help="Probe the configured MCP servers and report reachability + tools loaded.",
    )
    mcp_check.add_argument(
        "--timeout", type=float, default=20.0, help="Per-server connect timeout in seconds."
    )
    mcp_check.set_defaults(handler=cmd_mcp_check)

    return parser


def _add_agent_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interval-seconds", type=float, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--start-cycle", type=int, default=0)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--backoff-seconds", type=float, default=5.0)


def _load_cli_settings(args: argparse.Namespace) -> Settings:
    return load_settings(args.env_file)


def _home_from_args(args: argparse.Namespace, settings: Settings | None = None) -> Path:
    settings = settings or _load_cli_settings(args)
    return Path(args.home or settings.trading_agent_home or str(DEFAULT_HOME))


def _configure_cli_logging(args: argparse.Namespace, settings: Settings, home: Path) -> None:
    configure_logging(
        home,
        level=args.log_level or settings.log_level,
        log_to_stderr=settings.log_to_stderr,
        log_to_file=settings.log_to_file,
    )


def _require_testnet_exchange(settings: Settings) -> None:
    if settings.binance_venue != "testnet":
        raise RuntimeError("testnet-limit-order requires BINANCE_VENUE=testnet")
    if "testnet.binance.vision" not in settings.binance_api_base_url:
        raise RuntimeError("testnet-limit-order requires BINANCE_API_BASE_URL=https://testnet.binance.vision/api")


def _log_pnl_heartbeat(store: Store, config: Any, logger: Any) -> None:
    """Log realized + live-marked unrealized PnL so it refreshes between cycles."""
    try:
        positions = store.open_positions()
        prices = cached_current_prices(
            [o.symbol for o in positions], store, ttl=config.risk.mark_refresh_seconds
        )
        unrealized = 0.0
        marked = 0
        for order in positions:
            up, _ = unrealized_pnl(order, prices.get(order.symbol.upper()))
            if up is not None:
                unrealized += up
                marked += 1
        realized = store.summary().get("realized_pnl", 0.0)
        logger.info(
            "pnl heartbeat open_positions=%s marked=%s unrealized_usd=%.4f realized_usd=%s",
            len(positions),
            marked,
            unrealized,
            realized,
        )
    except Exception:
        logger.exception("pnl heartbeat failed")


def _sleep_with_pnl_heartbeat(
    store: Store, config: Any, logger: Any, sleep_seconds: float, runtime: Any | None = None
) -> None:
    """Sleep in short chunks between decision cycles, and on each chunk run the
    fast bracket monitor (TP/SL exits, no LLM) plus a PnL heartbeat. This is what
    makes a stop/take-profit touch act within ~bracket_monitor_seconds instead of
    waiting for the next hourly cycle. Also keeps stop requests responsive."""
    chunk_seconds = max(15.0, float(config.risk.bracket_monitor_seconds))
    slept = 0.0
    while slept < sleep_seconds:
        if store.get_setting("agent_stop_requested", False):
            return
        chunk = min(chunk_seconds, sleep_seconds - slept)
        time.sleep(chunk)
        slept += chunk
        if runtime is not None and not store.get_setting("kill_switch", False):
            try:
                runtime.monitor_open_positions()
            except Exception:
                logger.exception("bracket monitor tick failed")
        _log_pnl_heartbeat(store, config, logger)


def _duration_seconds(args: argparse.Namespace) -> float | None:
    provided = args.duration_hours is not None or args.duration_days is not None
    if not provided:
        return None
    seconds = 0.0
    if args.duration_hours is not None:
        if args.duration_hours < 0:
            raise SystemExit("--duration-hours must be >= 0")
        seconds += args.duration_hours * 3600
    if args.duration_days is not None:
        if args.duration_days < 0:
            raise SystemExit("--duration-days must be >= 0")
        seconds += args.duration_days * 86400
    return seconds


def cmd_init(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    _configure_cli_logging(args, settings, home)
    logger = get_logger("cli")
    logger.info("initializing trading agent home=%s", home)
    config = ensure_config(home)
    with Store(config.database_path) as store:
        store.set_setting("kill_switch", store.get_setting("kill_switch", False))
    print_json({"initialized": True, "home": str(home), "database": config.database_path, "environment": settings.redacted()})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    config = ensure_config(_home_from_args(args, settings))
    with Store(config.database_path) as store:
        payload = {
            "mode": config.mode,
            "decision_interval_minutes": config.decision_interval_minutes,
            "allowed_symbols": config.risk.allowed_symbols,
            "live": {
                "enabled": config.live.enabled,
                "venue_confirmed": config.live.venue_confirmed,
                "capital_budget_usd": config.live.capital_budget_usd,
            },
            "environment": settings.redacted(),
            "summary": store.summary(),
        }
    print_json(payload)
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    configure_logging(
        home,
        level=args.log_level or settings.log_level,
        log_to_stderr=False,  # REPL owns the terminal; logs go to file only
        log_to_file=True,
    )
    config = ensure_config(home)
    config.mode = settings.trading_agent_execution_mode
    save_config(config)
    return TradingAgentREPL(config, settings, symbols=args.symbols).run()


def cmd_exchange_ticker(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    _configure_cli_logging(args, settings, home)
    logger = get_logger("cli")
    adapter = BinanceSpotAdapter(base_url=settings.binance_api_base_url, settings=settings)
    logger.info("fetching ticker symbol=%s base_url=%s", args.symbol.upper(), settings.binance_api_base_url)
    print_json(adapter.ticker_price(args.symbol))
    return 0


def cmd_exchange_testnet_limit_order(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    _configure_cli_logging(args, settings, home)
    logger = get_logger("cli")
    _require_testnet_exchange(settings)
    credentials = BinanceSpotAdapter.credentials_from_env(settings=settings)
    adapter = BinanceSpotAdapter(base_url=settings.binance_api_base_url, settings=settings)
    action = "submit" if args.submit else "validate"
    logger.info(
        "starting Spot Testnet LIMIT order action=%s symbol=%s side=%s quantity=%s price=%s",
        action,
        args.symbol.upper(),
        args.side,
        args.quantity,
        args.price,
    )
    if args.submit:
        result = adapter.submit_limit_order(
            credentials,
            args.symbol,
            args.side,
            args.quantity,
            args.price,
            client_order_id=args.client_order_id,
        )
    else:
        result = adapter.validate_limit_order(
            credentials,
            args.symbol,
            args.side,
            args.quantity,
            args.price,
            client_order_id=args.client_order_id,
        )
    print_json(
        {
            "venue": settings.binance_venue,
            "base_url": settings.binance_api_base_url,
            "submitted": bool(args.submit),
            "symbol": args.symbol.upper(),
            "side": args.side,
            "quantity": args.quantity,
            "price": args.price,
            "result": result,
        }
    )
    return 0


def cmd_agent_once(args: argparse.Namespace) -> int:
    if args.cycle < 0:
        raise SystemExit("--cycle must be >= 0")
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    _configure_cli_logging(args, settings, home)
    logger = get_logger("cli")
    logger.info("starting supervised agent once cycle=%s symbols=%s", args.cycle, args.symbols or "default")
    config = ensure_config(home)
    config.mode = settings.trading_agent_execution_mode
    save_config(config)
    with Store(config.database_path) as store:
        result = SupervisorRuntime(config, store, settings=settings).run_once(
            cycle=args.cycle,
            symbols=args.symbols,
            thread_id=args.thread_id,
        )
        payload = {"result": asdict(result), "summary": store.summary()}
    print_json(payload)
    return 0


def cmd_agent_introduce(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    if settings.enable_llm_supervisor:
        require_model_api_key(settings)
    home = _home_from_args(args, settings)
    _configure_cli_logging(args, settings, home)
    logger = get_logger("cli")
    logger.info("starting agent introduction llm_supervisor=%s", settings.enable_llm_supervisor)
    config = ensure_config(home)
    config.mode = settings.trading_agent_execution_mode
    save_config(config)
    if settings.enable_llm_supervisor:
        with Store(config.database_path) as store:
            payload = SupervisorRuntime(config, store, settings=settings).introduce()
    else:
        payload = agent_introduction_payload(config, settings)
    print_json(payload)
    return 0


def cmd_agent_run(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    _configure_cli_logging(args, settings, home)
    logger = get_logger("cli")
    duration_seconds = _duration_seconds(args)
    logger.info(
        "starting supervised agent loop symbols=%s interval=%s max_cycles=%s duration_seconds=%s",
        args.symbols or "default",
        args.interval_seconds or "config",
        args.max_cycles,
        duration_seconds,
    )
    config = ensure_config(home)
    config.mode = settings.trading_agent_execution_mode
    save_config(config)
    owner = f"pid-{os.getpid()}-{new_id('lock')}"
    interval = (
        float(args.interval_seconds)
        if args.interval_seconds is not None
        else float(config.decision_interval_minutes * 60)
    )
    if interval < 0:
        raise SystemExit("--interval-seconds must be >= 0")
    if args.max_cycles is not None and args.max_cycles < 1:
        raise SystemExit("--max-cycles must be >= 1")
    if args.start_cycle < 0:
        raise SystemExit("--start-cycle must be >= 0")
    deadline = time.monotonic() + duration_seconds if duration_seconds is not None else None

    with Store(config.database_path) as store:
        if not store.try_acquire_agent_lock(owner):
            logger.warning("agent runner lock is already held")
            print_json({"started": False, "reason": "agent runner lock is already held", "summary": store.summary()})
            return 1
        store.set_setting("agent_stop_requested", False)
        store.log_event(
            "agent_loop_started",
            {
                "owner": owner,
                "symbols": args.symbols or config.risk.allowed_symbols,
                "interval_seconds": interval,
                "max_cycles": args.max_cycles,
                "duration_seconds": duration_seconds,
                "start_cycle": args.start_cycle,
            },
        )
        results: list[dict[str, Any]] = []
        failures = 0
        cycle = int(args.start_cycle)
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    logger.info("agent loop duration elapsed before cycle=%s", cycle)
                    break
                if store.get_setting("agent_stop_requested", False):
                    logger.info("agent stop requested before cycle=%s", cycle)
                    break
                if store.get_setting("kill_switch", False):
                    logger.warning("kill switch is enabled before cycle=%s", cycle)
                    store.log_event("agent_kill_switch_before_cycle", {"cycle": cycle})

                runtime = SupervisorRuntime(config, store, settings=settings)
                try:
                    result = runtime.run_once(cycle=cycle, symbols=args.symbols, thread_id=args.thread_id)
                    results.append(asdict(result))
                    failures = 0
                    logger.info(
                        "cycle completed cycle=%s run_id=%s intents=%s approved=%s rejected=%s submitted=%s errors=%s",
                        result.cycle,
                        result.run_id,
                        result.intent_count,
                        result.approved_count,
                        result.rejected_count,
                        result.submitted_trades,
                        result.error_count,
                    )
                except Exception as exc:
                    failures += 1
                    logger.exception("agent cycle failed cycle=%s consecutive_failures=%s", cycle, failures)
                    store.log_event(
                        "agent_cycle_error",
                        {"cycle": cycle, "error": str(exc), "consecutive_failures": failures},
                    )

                cycle += 1
                if args.max_cycles is not None and len(results) >= args.max_cycles:
                    break
                if store.get_setting("agent_stop_requested", False):
                    break
                sleep_seconds = interval if failures == 0 else min(args.backoff_seconds * (2 ** (failures - 1)), 300.0)
                if deadline is not None:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        logger.info("agent loop duration elapsed after cycle=%s", cycle - 1)
                        break
                    sleep_seconds = min(sleep_seconds, remaining_seconds)
                if sleep_seconds > 0:
                    logger.info("sleeping before next cycle seconds=%s", sleep_seconds)
                    _sleep_with_pnl_heartbeat(store, config, logger, sleep_seconds, runtime=runtime)
        finally:
            store.log_event(
                "agent_loop_finished",
                {"owner": owner, "completed_cycles": len(results), "next_cycle": cycle},
            )
            logger.info("releasing agent runner lock owner=%s", owner)
            store.release_agent_lock(owner)

        payload = {
            "results": results,
            "summary": store.summary(),
            "owner": owner,
            "duration_seconds": duration_seconds,
            "next_cycle": cycle,
        }
    print_json(payload)
    return 0


def cmd_agent_daemon(args: argparse.Namespace) -> int:
    return cmd_agent_run(args)


def cmd_agent_status(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    config = ensure_config(_home_from_args(args, settings))
    with Store(config.database_path) as store:
        payload = {
            "summary": store.summary(),
            "agent_stop_requested": bool(store.get_setting("agent_stop_requested", False)),
            "agent_lock": store.get_setting("agent_lock", None),
            "agent_checkpoint": store.get_setting("agent_checkpoint", None),
            "environment": settings.redacted(),
        }
    print_json(payload)
    return 0


def cmd_agent_stop(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    config = ensure_config(_home_from_args(args, settings))
    with Store(config.database_path) as store:
        store.set_setting("agent_stop_requested", True)
        store.log_event("agent_stop_requested", {"requested": True})
        payload = {"agent_stop_requested": True, "heartbeat": store.heartbeat()}
    print_json(payload)
    return 0


def cmd_agent_skills_list(args: argparse.Namespace) -> int:
    registry = BinanceSkillRegistry()
    payload = {
        "skills": [
            {
                "name": skill.name,
                "path": str(skill.path),
                "category": skill.category,
                "commands": list(skill.commands),
                "description": skill.description,
            }
            for skill in registry.list_skills()
        ],
        "active_readonly_sources": registry.read_only_skill_source_paths(),
    }
    print_json(payload)
    return 0


def cmd_agent_skills_show(args: argparse.Namespace) -> int:
    registry = BinanceSkillRegistry()
    skill = registry.skill(args.name)
    print(skill.skill_file.read_text(encoding="utf-8"))
    return 0


def cmd_agent_skills_commands(args: argparse.Namespace) -> int:
    registry = BinanceSkillRegistry()
    catalog = registry.command_catalog()
    if args.name:
        catalog = {args.name: catalog.get(args.name, [])}
    print_json({"commands": catalog})
    return 0


def cmd_agent_skills_run(args: argparse.Namespace) -> int:
    registry = BinanceSkillRegistry()
    result = registry.run_read_only_cli(args.name, args.skill_command, args.params_json)
    print_json(
        {
            "skill_name": result.skill_name,
            "command": result.command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
    return 0 if result.returncode == 0 else 1


def cmd_signals(args: argparse.Namespace) -> int:
    config = ensure_config(_home_from_args(args))
    with Store(config.database_path) as store:
        print_json({"signals": store.recent_evidence(args.limit)})
    return 0


def cmd_orders(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    config = ensure_config(_home_from_args(args, settings))
    with Store(config.database_path) as store:
        synced = 0
        if args.sync:
            reconciler = ExchangeReconciler(store, settings)
            if reconciler.available:
                synced = len(reconciler.reconcile())
        open_orders = store.open_positions()
        prices = cached_current_prices(
            [o.symbol for o in open_orders], store, ttl=config.risk.mark_refresh_seconds
        )
        open_marked: list[dict[str, Any]] = []
        unrealized_total = 0.0
        for order in open_orders:
            up, up_pct = unrealized_pnl(order, prices.get(order.symbol.upper()))
            if up is not None:
                unrealized_total += up
            open_marked.append(
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "status": order.status.value,
                    "entry_price": order.avg_fill_price or order.price,
                    "quantity": order.executed_qty or order.quantity,
                    "current_price": prices.get(order.symbol.upper()),
                    "unrealized_pnl_usd": up,
                    "unrealized_pnl_pct": up_pct,
                }
            )
        print_json(
            {
                "orders": store.all_orders(args.limit),
                "open_positions": open_marked,
                "unrealized_total_usd": round(unrealized_total, 8),
                "per_trade_pnl": store.per_trade_pnl(args.limit),
                "synced_orders": synced,
            }
        )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    settings = _load_cli_settings(args)
    home = _home_from_args(args, settings)
    log_path = home / "logs" / "trading_agent.log"
    if not log_path.exists():
        print_json({"error": f"no log file at {log_path}"})
        return 1

    def emit(line: str) -> None:
        # Log lines carry LLM/exchange Unicode; legacy Windows consoles are
        # cp1252 and plain print() dies on them. Replace, never crash.
        sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()

    for line in read_last_lines(log_path, args.lines):
        emit(line)
    if args.follow:
        try:
            for line in follow(log_path):
                emit(line)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_risk_config(args: argparse.Namespace) -> int:
    config = ensure_config(_home_from_args(args))
    print_json(asdict(config.risk))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    config = ensure_config(_home_from_args(args))
    symbols = args.symbols or config.risk.allowed_symbols
    adapter = BinanceSpotAdapter(base_url="https://api.binance.com/api")
    backtester = Backtester(config)
    summaries = []
    for symbol in symbols:
        klines = adapter.get_klines(symbol, interval=args.interval, limit=args.limit)
        summaries.append(backtester.run(symbol.upper(), klines).summary())
    print_json(
        {
            "interval": args.interval,
            "candles_requested": args.limit,
            "risk_config": asdict(config.risk),
            "results": summaries,
            "total_realized_pnl_usd": round(sum(s["realized_pnl_usd"] for s in summaries), 8),
        }
    )
    return 0


def _iso_to_epoch_ms(value: str) -> int | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def cmd_backtest_decisions(args: argparse.Namespace) -> int:
    config = ensure_config(_home_from_args(args))
    adapter = BinanceSpotAdapter(base_url="https://api.binance.com/api")
    with Store(config.database_path) as store:
        records = store.all_supervisor_decisions()

    windows: dict[str, list] = {}
    for rec in records:
        payload = rec.get("payload") or {}
        if str(payload.get("action") or "").upper() != "BUY":
            continue
        symbol = str(payload.get("symbol") or rec.get("symbol") or "")
        start_ms = _iso_to_epoch_ms(str(rec.get("created_at") or ""))
        if not symbol or start_ms is None:
            continue
        rows = adapter.get_klines(
            symbol, interval=args.interval, limit=args.window, start_time=start_ms
        )
        windows[str(rec.get("id"))] = candles_from_klines(rows)

    result = replay_recorded_decisions(
        records, windows, config, entry_ttl_candles=args.entry_ttl
    )
    output = result.summary()
    output["interval"] = args.interval
    output["forward_window_candles"] = args.window
    print_json(output)
    return 0


def cmd_mcp_check(args: argparse.Namespace) -> int:
    home = str(_home_from_args(args))
    servers = load_mcp_config(home)
    enabled = [s for s in servers if s.enabled and s.transport != "stdio"]
    report: dict[str, Any] = {
        "configured": len(servers),
        "enabled": len(enabled),
        "servers": [],
    }
    # Probe each enabled server on its own so one failure is isolated and named.
    for server in enabled:
        result = MCPToolLoader([server], timeout_seconds=args.timeout).load_tools_sync()
        report["servers"].append(
            {
                "name": server.name,
                "url": server.url,
                "reachable": result.server_count > 0,
                "tools_loaded": len(result.tools),
                "tool_names": [getattr(t, "name", "?") for t in result.tools][:25],
                "blocked_execution_tools": result.blocked_tool_count,
                "errors": result.errors,
            }
        )
    report["disabled"] = [s.name for s in servers if not s.enabled]
    print_json(report)
    reachable = sum(1 for s in report["servers"] if s["reachable"])
    return 0 if reachable == len(enabled) else 1


def cmd_kill_switch(args: argparse.Namespace) -> int:
    config = ensure_config(_home_from_args(args))
    enabled = args.state == "on"
    with Store(config.database_path) as store:
        store.set_setting("kill_switch", enabled)
        store.log_event("kill_switch_changed", {"enabled": enabled})
    print_json({"kill_switch": enabled})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = ensure_config(_home_from_args(args))
    with Store(config.database_path) as store:
        prices = current_prices([order.symbol for order in store.open_positions()])
        output = (
            report_json(store, config, prices=prices)
            if args.format == "json"
            else report_markdown(store, config, prices=prices)
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print_json({"written": str(path)})
    else:
        print(output)
    return 0


# Options owned by the top-level parser; they must precede the subcommand.
_GLOBAL_OPTIONS = {"--home", "--env-file", "--log-level"}


def _split_global_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Hoist top-level options out of a flat argv so a subcommand can be injected."""
    global_args: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.split("=", 1)[0] in _GLOBAL_OPTIONS:
            global_args.append(arg)
            if "=" not in arg and index + 1 < len(argv):
                index += 1
                global_args.append(argv[index])
        else:
            rest.append(arg)
        index += 1
    return global_args, rest


def repl_main() -> int:
    """Console-script entry point for trading-agent-repl.

    argparse only accepts --home/--env-file/--log-level before the subcommand,
    so `trading-agent-repl --env-file .env` must become
    `trading-agent --env-file .env repl`, not `trading-agent repl --env-file .env`.
    """
    global_args, repl_args = _split_global_args(list(sys.argv[1:]))
    return main([*global_args, "repl", *repl_args])


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
