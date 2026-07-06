"""Shared state types for the supervised trading cycle graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

EventCallback = Callable[[str, str, dict[str, Any]], None]


class RuntimeGraphState(TypedDict, total=False):
    # Inputs
    cycle: int
    symbols: list[str]
    thread_id: str
    # prepare_context
    run_id: str
    mcp_tool_count: int
    errors: list[str]
    snapshots: dict[str, Any]  # symbol -> MarketSnapshot
    evidence: list[Any]  # EvidenceRecord
    baseline_intents: list[Any]  # TradeIntent from the deterministic strategy
    level_maps: dict[str, Any]  # symbol -> LevelMap (support/resistance zones + regime)
    position_reviews: list[Any]  # PositionReview per open order (manage risk first)
    cycle_tier: str  # FULL | REVIEW | SKIP (material-change cost tier)
    # consult_agents / parse_decision
    supervisor_message: str
    decisions: list[Any]  # SupervisorDecision
    # risk_gate
    approved: list[Any]  # (decision, intent-or-None) pairs ready to execute
    approved_count: int
    rejected_count: int
    # execute
    submitted_orders: list[str]
    reconciled_count: int
    # report
    result: dict[str, Any]


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    thread_id: str
    cycle: int
    evidence_count: int
    intent_count: int
    approved_count: int
    rejected_count: int
    submitted_trades: int
    reconciled_trades: int
    mcp_tool_count: int
    error_count: int
    decision_count: int = 0
    wait_count: int = 0
