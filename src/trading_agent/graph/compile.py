"""Graph assembly for the supervised trading cycle.

The cycle graph is compiled WITHOUT a checkpointer on purpose: every run is a
fresh deterministic pipeline, and its state holds live Python objects
(snapshots, tool handles) that must never be serialized. Conversation memory
lives in the deep agent, which gets the persistent SQLite checkpointer (see
checkpointer.py) under the operator's thread id; sharing one checkpointer
thread between the outer graph and the nested deep agent corrupts both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import StateGraph

from trading_agent.graph.edges import wire_edges
from trading_agent.graph.state import RuntimeGraphState

if TYPE_CHECKING:
    from trading_agent.graph.nodes import CycleNodes


def build_cycle_graph(nodes: CycleNodes) -> Any:
    graph = StateGraph(RuntimeGraphState)
    graph.add_node("prepare_context", nodes.prepare_context)
    graph.add_node("consult_agents", nodes.consult_agents)
    graph.add_node("parse_decision", nodes.parse_decision)
    graph.add_node("risk_gate", nodes.risk_gate)
    graph.add_node("execute", nodes.execute)
    graph.add_node("report", nodes.report)
    wire_edges(graph)
    return graph.compile()
