"""Post-trade reflection: turn each closed round trip into a stored lesson.

Layered-memory pattern (FinMem/FinAgent): a desk only improves if outcomes feed
back into future decisions. On close we compute deterministic stats (realized R,
outcome, holding time) and a short lesson, persist it, and the cycle context
surfaces recent reflections + aggregate trade_stats so the supervisor can see
"what my recent trades actually did" instead of deciding from a blank slate.

This is deterministic by design (no LLM call): it must never fail or add cost on
the close path. The text lesson is a fixed template over the closed order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from trading_agent.core.models import OrderRecord, new_id, utc_iso


@dataclass(slots=True)
class Reflection:
    symbol: str
    outcome: str  # "win" | "loss" | "scratch"
    realized_pnl: float
    realized_r: float | None
    exit_reason: str | None
    holding_minutes: float | None
    lesson: str
    order_id: str | None = None
    created_at: str = field(default_factory=utc_iso)
    id: str = field(default_factory=lambda: new_id("refl"))


def _holding_minutes(order: OrderRecord) -> float | None:
    if not order.opened_at or not order.closed_at:
        return None
    try:
        opened = datetime.fromisoformat(order.opened_at)
        closed = datetime.fromisoformat(order.closed_at)
    except (TypeError, ValueError):
        return None
    return round((closed - opened).total_seconds() / 60.0, 2)


def realized_r(order: OrderRecord) -> float | None:
    """Realized profit in units of the INITIAL risk (entry -> initial stop).

    R-multiple is how a desk actually judges a trade: +2R is a good win, -1R a
    clean stop. Returns None when the initial risk is undefined (no stop or a
    non-long/zero-quantity order)."""
    entry = order.avg_fill_price or order.price
    quantity = order.executed_qty or order.quantity
    initial_stop = (
        order.exit_plan.initial_stop_price if order.exit_plan is not None else order.stop_loss_price
    )
    risk_per_unit = entry - initial_stop
    if risk_per_unit <= 0 or quantity <= 0:
        return None
    return round(order.realized_pnl / (risk_per_unit * quantity), 3)


def build_reflection(order: OrderRecord) -> Reflection:
    pnl = round(order.realized_pnl, 8)
    outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "scratch"
    r_multiple = realized_r(order)
    held = _holding_minutes(order)
    reason = order.exit_reason or order.closed_by or "closed"
    r_text = f"{r_multiple:+.2f}R " if r_multiple is not None else ""
    held_text = f"after {held:.0f}m " if held is not None else ""
    lesson = f"{order.symbol} closed {r_text}({pnl:+.2f}) via {reason} {held_text}- {outcome}."
    return Reflection(
        symbol=order.symbol,
        outcome=outcome,
        realized_pnl=pnl,
        realized_r=r_multiple,
        exit_reason=reason,
        holding_minutes=held,
        lesson=lesson,
        order_id=order.id,
    )
