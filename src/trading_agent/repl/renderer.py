from __future__ import annotations

import datetime
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trading_agent.repl.events import AgentEvent

_AGENT_STYLES = {
    "supervisor": "bold magenta",
    "market_research": "cyan",
    "news_research": "yellow",
    "onchain_research": "green",
    "strategy": "bold blue",
    "risk_review": "red",
    "risk_governor": "bold red",
    "reporting": "white",
    "market_feed": "cyan",
    "market_data_agent": "cyan",
    "news_sentiment_agent": "yellow",
    "onchain_flow_agent": "green",
    "execution": "bold green",
    "lifecycle": "bold white",
}

# ASCII-only glyphs: piped stdout on Windows may be cp1252-encoded.
_KIND_GLYPHS = {
    "lifecycle": "@",
    "supervisor": ">",
    "token": ".",
    "update": "->",
    "snapshot": "$",
    "evidence": "=",
    "proposal": "*",
    "risk_decision": "#",
    "decision": "%",
    "order": "^",
    "reconcile": "~",
    "skill_call": "+",
    "error": "!",
    "info": "i",
}

_HIDDEN_TOKEN_AGENTS = {
    "tools",
    "tool",
    "SkillsMiddleware.before_agent",
    "PatchToolCallsMiddleware.before_agent",
    "TodoListMiddleware.after_model",
}


class AgentRenderer:
    """Formats agent events into a Claude Code-style scrolling stream."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._token_buffer: dict[str, str] = {}

    def render_event(self, event: AgentEvent) -> None:
        if event.kind == "token":
            if event.agent in _HIDDEN_TOKEN_AGENTS:
                return
            self._render_token(event)
            return
        self.flush_tokens()
        line = self._format_line(event)
        if line is not None:
            self.console.print(line)

    def flush_tokens(self) -> None:
        for agent, text in self._token_buffer.items():
            if text.strip():
                style = _AGENT_STYLES.get(agent, "white")
                prefix = Text(f"[{agent:<17}] ", style=style)
                self.console.print(prefix + Text(text.strip()[:500], style="dim"))
        self._token_buffer.clear()

    def _render_token(self, event: AgentEvent) -> None:
        text = str(event.data.get("text", ""))
        buffered = self._token_buffer.get(event.agent, "") + text
        # Flush at sentence boundaries to keep the stream readable.
        if len(buffered) > 400 or buffered.endswith((".", "!", "?", "\n")):
            self._token_buffer[event.agent] = buffered
            self.flush_tokens()
        else:
            self._token_buffer[event.agent] = buffered

    def _format_line(self, event: AgentEvent) -> Text | None:
        glyph = _KIND_GLYPHS.get(event.kind, "*")
        style = _AGENT_STYLES.get(event.agent, "white")
        stamp = datetime.datetime.fromtimestamp(event.ts).strftime("%H:%M:%S")
        head = Text(f"{stamp} ", style="dim") + Text(f"[{event.agent:<17}] {glyph} ", style=style)
        body = self._format_body(event)
        if body is None:
            return None
        return head + body

    def _format_body(self, event: AgentEvent) -> Text | None:
        data = event.data
        if event.kind == "lifecycle":
            return Text(f"state {data.get('from')} -> {data.get('to')}", style="bold")
        if event.kind == "supervisor":
            return Text(str(data.get("message", "")), style="magenta")
        if event.kind == "update":
            summary = str(data.get("summary", "")).strip()
            if not summary:
                return None
            return Text(f"{data.get('node', '')}: {summary[:240]}")
        if event.kind == "snapshot":
            source = data.get("source", "?")
            source_style = "bold green" if source != "simulated" else "yellow"
            return (
                Text(f"{data.get('symbol')} ${_fmt_price(data.get('last_price'))} ")
                + Text(f"[{source}]", style=source_style)
            )
        if event.kind == "evidence":
            return Text(
                f"{data.get('symbol')} {data.get('kind')} score={data.get('score'):+.3f} "
                f"conf={data.get('confidence'):.2f} src={data.get('source')}"
            )
        if event.kind == "proposal":
            return Text(
                f"PROPOSAL {data.get('side')} {data.get('symbol')} @ {_fmt_price(data.get('limit_price'))} "
                f"qty={data.get('quantity')} conf={data.get('confidence'):.2f} "
                f"edge={data.get('expected_edge_bps'):.1f}bps",
                style="bold blue",
            )
        if event.kind == "decision":
            rationale = str(data.get("rationale", ""))
            return Text(
                f"DECISION {data.get('action')} {data.get('symbol')} "
                f"conf={data.get('confidence', 0):.2f} {rationale[:160]}",
                style="bold magenta",
            )
        if event.kind == "risk_decision":
            if data.get("approved"):
                return Text(f"APPROVED {data.get('symbol')}", style="bold green")
            reasons = "; ".join(data.get("reasons", []))
            return Text(f"REJECTED {data.get('symbol')}: {reasons}", style="bold red")
        if event.kind == "order":
            if data.get("sync"):
                spent = data.get("quote_spent")
                spent_note = f" spent={_fmt_price(spent)}" if spent else ""
                return Text(
                    f"{str(data.get('mode', '')).upper()} {data.get('symbol')} "
                    f"{data.get('status')} exchange={data.get('exchange_status')} "
                    f"filled={data.get('executed_qty')}{spent_note}",
                    style="bold green",
                )
            return Text(
                f"{str(data.get('mode', '')).upper()} order {data.get('symbol')} "
                f"@ {_fmt_price(data.get('price'))} qty={data.get('quantity')}",
                style="bold green",
            )
        if event.kind == "reconcile":
            return Text(f"reconciled {data.get('count')} position(s)")
        if event.kind == "skill_call":
            return Text(f"{data.get('skill')}/{data.get('command')} {data.get('status', '')}", style="dim")
        if event.kind == "error":
            return Text(str(data.get("message", "")), style="bold red")
        return Text(str(data.get("message", data))[:240])

    def render_markdown(self, agent: str, text: str, *, title: str | None = None) -> None:
        """Markdown preview in the terminal: agent replies render as styled
        panels instead of raw text."""
        self.flush_tokens()
        style = _AGENT_STYLES.get(agent, "white")
        self.console.print(
            Panel(Markdown(text), title=title or agent, border_style=style, expand=False)
        )

    def render_header(self, status: dict[str, Any]) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        for key, value in status.items():
            table.add_row(str(key), str(value))
        self.console.print(Panel(table, title="Trading Agent", border_style="cyan"))

    def render_help(self, commands: dict[str, str]) -> None:
        table = Table(title="Commands", show_header=False, border_style="dim")
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for command, description in commands.items():
            table.add_row(command, description)
        self.console.print(table)


def _fmt_price(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 1000:
        return f"{number:,.2f}"
    return f"{number:.4f}"
