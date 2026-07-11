"""Deep-agent construction: supervisor, subagents, skills, and prompts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend

from trading_agent.core.config import Settings
from trading_agent.core.decision import DECISION_FORMAT_REFERENCE
from trading_agent.prompts import prompt_for
from trading_agent.utils.binance_skills import BinanceSkillRegistry, run_binance_research_cli
from trading_agent.utils.market_data import MARKET_DATA_TOOLS
from trading_agent.utils.web_search import WEB_RESEARCH_TOOLS

# Skill files larger than this are not seeded into agent state.
_MAX_SKILL_FILE_BYTES = 100_000

# Subagent names that may carry a per-agent model override in config.
SUBAGENT_NAMES = (
    "market_research",
    "technical_analyst",
    "news_research",
    "onchain_research",
    "bull_researcher",
    "bear_researcher",
    "strategy",
    "risk_review",
    "reporting",
)


def binance_research_tools() -> list[Any]:
    return [run_binance_research_cli]


def base_research_tools() -> list[Any]:
    return binance_research_tools() + list(WEB_RESEARCH_TOOLS) + list(MARKET_DATA_TOOLS)


def subagent_specs(
    *,
    tools: list[Any] | None = None,
    skills: list[str],
    subagent_models: dict[str, Any] | None = None,
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Specs for the six specialists. Each may carry its own `model` — a
    "provider:model" string OR a pre-built chat-model instance (deepagents'
    resolve_model accepts both; an instance is used for Azure, whose string
    resolution needs env vars we set on Settings instead). Without one it inherits
    the supervisor's model. ``only`` returns just the named subset (REVIEW tier)."""
    research_tools = list(tools or []) + base_research_tools()
    models = subagent_models or {}

    def spec(name: str, description: str, agent_tools: list[Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": name,
            "description": description,
            "system_prompt": prompt_for(name).text,
            "tools": agent_tools,
            "skills": skills,
        }
        if models.get(name):
            entry["model"] = models[name]
        return entry

    analysis_tools = list(MARKET_DATA_TOOLS)
    specs = [
        spec(
            "market_research",
            "Analyze OHLCV, klines/indicators, spread, volume, liquidity, and order-book evidence.",
            research_tools,
        ),
        spec(
            "technical_analyst",
            "Confirm the real demand/support zone to bid and the invalidation level, "
            "from the computed level_maps + multi-timeframe candles (monthly..1h).",
            analysis_tools,
        ),
        spec(
            "news_research",
            "Analyze news, announcements, and social sources; share article URLs for follow-up.",
            research_tools,
        ),
        spec(
            "onchain_research",
            "Analyze read-only wallet, token, exchange-flow, and protocol evidence; visit URLs from news_research.",
            research_tools,
        ),
        spec(
            "bull_researcher",
            "Argue the strongest LONG case from the gathered evidence, with explicit invalidation.",
            analysis_tools,
        ),
        spec(
            "bear_researcher",
            "Argue the strongest SHORT/avoid case from the gathered evidence, with explicit invalidation.",
            analysis_tools,
        ),
        spec(
            "strategy",
            "Turn evidence into structured trade stances with confidence and expected edge.",
            analysis_tools,
        ),
        spec(
            "risk_review",
            "Critique proposed actions without overriding deterministic risk checks.",
            analysis_tools,
        ),
        spec(
            "reporting",
            "Summarize logs, per-trade PnL, rejections, and open-position status.",
            # Reporting needs the injected ops tools (open orders, order status,
            # per-trade PnL, recent decisions) to do its job.
            list(tools or []),
        ),
    ]
    if only is not None:
        specs = [entry for entry in specs if entry["name"] in only]
    return specs


def build_deep_agent(
    *,
    model: Any,
    subagents: list[dict[str, Any]],
    skills: list[str],
    checkpointer: Any,
    extra_tools: list[Any] | None = None,
) -> Any:
    # StateBackend keeps the agent filesystem in LangGraph state: it is the
    # only backend whose skill loading works on Windows (deepagents #889),
    # and it sandboxes agents away from the real disk entirely. Skill files
    # are seeded into state per invocation via skill_state_files().
    return create_deep_agent(
        model=model,
        tools=base_research_tools() + list(extra_tools or []),
        system_prompt=prompt_for("supervisor").text,
        subagents=subagents,
        skills=skills,
        backend=StateBackend(),
        checkpointer=checkpointer,
        name="trading-supervisor",
    )


def skill_source_paths(project_root: Path, binance_skills: BinanceSkillRegistry) -> list[str]:
    sources: list[str] = []
    project_skills = project_root / "skills"
    if project_skills.exists():
        sources.append("/skills")
    for source in binance_skills.read_only_skill_source_paths():
        if (project_root / source).exists():
            sources.append(f"/{source}")
    return sources


def skill_state_files(project_root: Path, sources: list[str]) -> dict[str, dict[str, str]]:
    """Read approved skill sources from disk into in-state FileData entries.

    pathlib handles Windows paths correctly; the in-state paths always use
    forward slashes so SkillsMiddleware can discover them via StateBackend.
    Values follow deepagents' FileData shape: {"content", "encoding"} PLUS
    ``created_at`` / ``modified_at`` ISO timestamps — deepagents' glob reads
    ``file_data["modified_at"]`` with no default, so a seeded file lacking it
    raised KeyError and crashed the whole cycle when an agent globbed it.
    """
    files: dict[str, dict[str, str]] = {}
    for source in sources:
        root = project_root / source.lstrip("/")
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > _MAX_SKILL_FILE_BYTES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            state_path = "/" + path.relative_to(project_root).as_posix()
            timestamp = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
            files[state_path] = {
                "content": content,
                "encoding": "utf-8",
                "created_at": timestamp,
                "modified_at": timestamp,
            }
    return files


def mode_description(settings: Settings) -> str:
    mode = settings.trading_agent_execution_mode
    if mode == "live":
        return (
            "LIVE trading on Binance spot with real funds. Orders are real money: "
            "deterministic risk checks and operator confirmation gate every order."
        )
    if mode == "testnet":
        return (
            "Binance Spot Testnet trading: real limit orders against "
            "testnet.binance.vision with test funds. Order status and PnL are "
            "reconciled from live exchange fills."
        )
    return "no exchange execution configured; set testnet or live before running agents."


def build_intro_prompt(settings: Settings, symbols: list[str]) -> str:
    return (
        "Introduce this trading-agent service as the supervisor deep agent. "
        "Have the supervisor and specialist subagents introduce their roles. "
        f"Current execution mode: {mode_description(settings)} "
        "Explain that the supervisor makes the final trade decisions "
        "(BUY/SELL/WAIT/CLOSE/ADJUST) after consulting every specialist, and "
        "that a deterministic risk gate can veto any decision. "
        f"Ask the operator which spot symbols to trade. Defaults: {', '.join(symbols)}. "
        "Keep the response concise."
        + ("" if settings.trading_agent_execution_mode == "live" else " Do not suggest enabling live trading.")
    )


def build_chat_prompt(message: str, context: dict[str, Any]) -> str:
    return (
        "OPERATOR CHAT. The human operator is talking to you directly. Answer "
        "their question or carry out their request using your tools and "
        "subagents. Format your reply as clean markdown for a terminal.\n\n"
        "CURRENT ACCOUNT CONTEXT:\n"
        f"{json.dumps(context, indent=2, sort_keys=True, default=str)}\n\n"
        "Rules:\n"
        "- For any current price, call get_price/get_orderbook_ticker; for chart "
        "questions, call get_klines. Never answer price questions from memory.\n"
        "- Use list_open_orders/get_order_status/recent_trades_pnl for order and "
        "PnL questions; exchange status beats the local cache. Use "
        "recent_decisions to recall what was decided or proposed in past cycles "
        "and why the gate approved/rejected it.\n"
        "- If (and only if) the operator asks you to open, close, or adjust a "
        "position, end your reply with the standard fenced json decision block "
        f"(actions BUY/SELL/CLOSE/ADJUST as defined below). An explicit operator "
        "instruction is AUTHORITATIVE: it overrides desk policy, including the "
        "no-touch rule for open positions. Do not refuse or substitute WAIT; you "
        "may note concerns in prose, but emit the requested decision block. The "
        "operator will be asked to confirm, and the deterministic risk gate still "
        "applies. For a BUY you must consult all seven subagents first.\n"
        "- Otherwise do NOT emit a decision block.\n\n"
        f"{DECISION_FORMAT_REFERENCE}\n\n"
        f"OPERATOR MESSAGE:\n{message}"
    )


def build_cycle_prompt(
    cycle: int, symbols: list[str], context: dict[str, Any], *, tier: str = "FULL"
) -> str:
    now = context.get("now", {}) if isinstance(context, dict) else {}
    now_line = ""
    if now:
        now_line = (
            f"CURRENT TIME: today is {now.get('weekday')}, {now.get('date')} "
            f"{now.get('time_utc')} UTC. Anchor every news/search query and data "
            f"window to this moment (e.g. last 24h from now); never assume an older date.\n\n"
        )
    if tier == "REVIEW":
        # NEWS SENTRY: positions are mechanical (no-touch); the only job is
        # scanning for a critical adverse catalyst on held symbols. Minimal tokens.
        return (
            f"NEWS SENTRY cycle {cycle} for {', '.join(symbols)} — held-symbol news check ONLY.\n\n"
            f"{now_line}"
            "Open positions are managed MECHANICALLY: the exit ladder banks TP tiers, trails the "
            "runner, and the stop protects downside. You do NOT manage them. Do NOT open, adjust, "
            "or close anything on price action, PnL, or chart structure — a position near its stop "
            "or take-profit needs NO action from you (the ladder handles it).\n"
            "Your ONLY task: for each symbol with an open position or resting bid in "
            "`position_reviews`, check the last 24h of news (get_crypto_news first; "
            "web_news_search to fill gaps) for a CRITICAL adverse catalyst — exchange/protocol "
            "hack, delisting, regulatory enforcement with direct impact, insolvency. Routine "
            "bearish sentiment, price commentary, or macro chop is NOT critical.\n"
            "- No critical news (the normal case): emit WAIT for every symbol, one line each.\n"
            "- Critical adverse catalyst on a HELD symbol: CLOSE that order_id, citing the source "
            "URL in the rationale.\n"
            "Keep the whole reply brief.\n\n"
            "CYCLE CONTEXT (account + open positions + position_reviews):\n"
            f"{json.dumps(context, indent=2, sort_keys=True, default=str)}\n\n"
            f"{DECISION_FORMAT_REFERENCE}"
        )
    return (
        f"Run one supervised trading cycle for symbols {', '.join(symbols)} at cycle {cycle}.\n\n"
        f"{now_line}"
        "Work in TWO ORDERED PHASES. Review the CYCLE CONTEXT first:\n"
        f"{json.dumps(context, indent=2, sort_keys=True, default=str)}\n\n"
        "PHASE 1 — OPEN ORDERS ARE NO-TOUCH; NEWS-RISK CHECK ONLY. Filled positions are "
        "managed MECHANICALLY: the exit ladder banks TP tiers (3%/6%/10%), trails the runner, "
        "and the stop is the only exit. Do NOT close, sell, or adjust a position on price "
        "action, PnL, or chart structure — being near a stop or target needs no action. The "
        "SINGLE exception: have news_research check held symbols for a CRITICAL adverse "
        "catalyst (hack, delisting, regulatory enforcement, insolvency — not routine bearish "
        "sentiment); only then CLOSE that order_id, citing the source URL. A RESTING bid stays "
        "working until filled or TTL (~3 days) — KEEP it; cancel (CLOSE its order_id) only if "
        "its `position_reviews` entry flags CANCEL_CANDIDATE (regime flipped) or a critical "
        "catalyst hit that symbol.\n"
        "PHASE 2 — NEW ENTRIES (only if capacity remains). Symbols that already have an open "
        "position or resting bid are NOT new-BUY candidates. Only if open slots remain (position "
        "cap + correlation budget), consult ALL seven subagents (market_research, "
        "technical_analyst, news_research, onchain_research, strategy, risk_review, reporting) "
        "and, using `level_maps` (computed demand/supply zones + regime), have technical_analyst "
        "confirm the support zone to bid. Prefer a resting demand-zone BUY over WAIT when the "
        "read is not bearish. If there is no capacity, skip Phase 2 entirely.\n"
        "Use get_price/get_orderbook_ticker/get_klines for every price you cite — never infer "
        "prices from news or memory.\n\n"
        f"{DECISION_FORMAT_REFERENCE}"
    )
