"""Operator /chat flow: talk to the supervisor team, confirm trade actions.

Flow per message:
1. runtime.chat() -> supervisor reply (rendered as terminal markdown)
2. any BUY/SELL/CLOSE/ADJUST decision block in the reply is summarized
3. operator confirms each action explicitly (y/N)
4. confirmed decisions run the SAME deterministic gate + execution path as
   supervised cycles (runtime.execute_operator_decision)
"""

from __future__ import annotations

from typing import Any, Callable

from rich.console import Console

from trading_agent.core.decision import DecisionAction, SupervisorDecision
from trading_agent.core.logging import get_logger
from trading_agent.graph import SupervisorRuntime
from trading_agent.repl.renderer import AgentRenderer

LOGGER = get_logger("repl.chat")

ConfirmFn = Callable[[str], bool]


def handle_chat(
    runtime: SupervisorRuntime,
    renderer: AgentRenderer,
    console: Console,
    message: str,
    *,
    confirm: ConfirmFn,
) -> None:
    result = runtime.chat(message)
    renderer.render_markdown("supervisor", result["message"], title="supervisor /chat")
    for problem in result.get("errors", []):
        console.print(f"[yellow]decision parse note: {problem}[/yellow]")
    for decision in result.get("decisions", []):
        _handle_decision(runtime, console, decision, confirm)


def _handle_decision(
    runtime: SupervisorRuntime,
    console: Console,
    decision: SupervisorDecision,
    confirm: ConfirmFn,
) -> None:
    summary = _describe(decision)
    if not confirm(f"Execute {summary}? [y/N] "):
        console.print(f"[yellow]skipped: {summary}[/yellow]")
        runtime.store.save_supervisor_decision(
            decision_id=decision.id,
            run_id="operator_chat",
            action=decision.action.value,
            symbol=decision.symbol,
            payload=decision.summary_payload(),
            gate_approved=False,
            gate_reasons=["operator declined"],
            source=decision.source,
        )
        return
    try:
        outcome: dict[str, Any] = runtime.execute_operator_decision(decision)
    except Exception as exc:
        LOGGER.exception("chat decision execution failed decision_id=%s", decision.id)
        console.print(f"[bold red]execution failed:[/bold red] {exc}")
        return
    if outcome.get("executed"):
        console.print(
            f"[bold green]executed[/bold green] {summary} (order {outcome.get('order_id')})"
        )
    else:
        console.print(f"[bold red]not executed:[/bold red] {outcome.get('reason')}")


def _describe(decision: SupervisorDecision) -> str:
    if decision.action == DecisionAction.BUY:
        return (
            f"BUY {decision.quantity:g} {decision.symbol} @ {decision.limit_price:g} "
            f"(conf {decision.confidence:.2f})"
        )
    if decision.action in {DecisionAction.CLOSE, DecisionAction.SELL}:
        return f"{decision.action.value} {decision.symbol} order {decision.target_order_id}"
    if decision.action == DecisionAction.ADJUST:
        parts = []
        if decision.new_take_profit_price is not None:
            parts.append(f"TP->{decision.new_take_profit_price:g}")
        if decision.new_stop_loss_price is not None:
            parts.append(f"SL->{decision.new_stop_loss_price:g}")
        return f"ADJUST {decision.symbol} order {decision.target_order_id} ({', '.join(parts)})"
    return f"{decision.action.value} {decision.symbol}"
