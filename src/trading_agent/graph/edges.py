"""Edge wiring and conditional routers for the trading cycle graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from trading_agent.core.decision import DecisionAction
from trading_agent.graph.state import RuntimeGraphState


def route_after_prepare(state: RuntimeGraphState) -> str:
    """SKIP cycles bypass the expensive deep agent and go straight to the
    deterministic decision (all-WAIT); REVIEW/FULL consult the supervisor."""
    if state.get("cycle_tier") == "SKIP":
        return "parse_decision"
    return "consult_agents"


def route_after_parse(state: RuntimeGraphState) -> str:
    """WAIT-only cycles skip the gate entirely and go straight to reporting."""
    decisions = state.get("decisions", [])
    if any(d.action != DecisionAction.WAIT for d in decisions):
        return "risk_gate"
    return "report"


def route_after_gate(state: RuntimeGraphState) -> str:
    """Nothing approved means nothing to execute."""
    if state.get("approved"):
        return "execute"
    return "report"


def wire_edges(graph: StateGraph) -> StateGraph:
    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges(
        "prepare_context",
        route_after_prepare,
        {"consult_agents": "consult_agents", "parse_decision": "parse_decision"},
    )
    graph.add_edge("consult_agents", "parse_decision")
    graph.add_conditional_edges(
        "parse_decision", route_after_parse, {"risk_gate": "risk_gate", "report": "report"}
    )
    graph.add_conditional_edges(
        "risk_gate", route_after_gate, {"execute": "execute", "report": "report"}
    )
    graph.add_edge("execute", "report")
    graph.add_edge("report", END)
    return graph
