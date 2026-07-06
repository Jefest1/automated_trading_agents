from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from trading_agent.core.config import AppConfig, Settings


@dataclass(frozen=True, slots=True)
class AgentIntroduction:
    name: str
    role: str
    first_message: str


def agent_introduction_payload(config: AppConfig, settings: Settings) -> dict[str, Any]:
    agents = [
        AgentIntroduction(
            name="supervisor",
            role="Coordinates the LangGraph/Deep Agents cycle and delegates research.",
            first_message="I coordinate the research agents, maintain the cycle context, and never execute trades.",
        ),
        AgentIntroduction(
            name="market_research",
            role="Reads market structure evidence for spot symbols.",
            first_message="I watch OHLCV, volume, spread, and liquidity evidence for the selected tokens.",
        ),
        AgentIntroduction(
            name="news_research",
            role="Reads free-first news and announcement evidence.",
            first_message="I check GDELT, RSS-style feeds, Binance announcements, and source quality.",
        ),
        AgentIntroduction(
            name="onchain_research",
            role="Reads public on-chain and token-flow evidence.",
            first_message="I use approved read-only sources and Binance research skills for token and flow context.",
        ),
        AgentIntroduction(
            name="strategy",
            role="Turns evidence into flat spot TradeIntent proposals.",
            first_message="I can propose a testnet/live spot entry with TP/SL assumptions, but I cannot place orders.",
        ),
        AgentIntroduction(
            name="risk_review",
            role="Critiques proposals before deterministic risk checks.",
            first_message="I flag missing evidence, stale data, weak scores, and cap conflicts without overriding RiskGovernor.",
        ),
        AgentIntroduction(
            name="reporting",
            role="Summarizes logs, PnL, rejections, and promotion status.",
            first_message="I keep the audit trail readable so we can evaluate testnet execution quality.",
        ),
    ]
    return {
        "source": "deterministic",
        "mode": config.mode,
        "default_symbols": list(config.risk.allowed_symbols),
        "agents": [asdict(agent) for agent in agents],
        "environment": settings.redacted(),
        "question": (
            "Which tokens should we start testnet trading with? "
            "Use spot symbols such as BTCUSDT ETHUSDT SOLUSDT BNBUSDT."
        ),
        "next_commands": [
            "uv run python -m trading_agent.cli agent once --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT",
            "uv run python -m trading_agent.cli agent run --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT",
        ],
    }
