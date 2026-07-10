"""Deterministic open-order review: the position-management counterpart to the
entry baseline (agents/strategy.py).

Each cycle, BEFORE any new-entry research, the desk reviews open risk first. This
agent turns every open order; a filled position or a resting (unfilled) bid; into
a conservative recommended action the LLM team then confirms or overrides:

- filled position  -> HOLD by default (the tiered exit ladder owns the stop); flag
  CLOSE_CANDIDATE only on a hard deterministic invalidation (confirmed downtrend +
  losing + past the min-hold cooldown).
- resting bid      -> KEEP while within its TTL; flag CANCEL_CANDIDATE if the regime
  flipped against the bid's bullish thesis.

It never proposes loosening a stop and never forces an exit; the deterministic
stop-loss/exit-ladder protects downside regardless, and the bid TTL is the backstop
for unfilled bids.
"""

from __future__ import annotations

from datetime import UTC, datetime

from trading_agent.core.config import AppConfig
from trading_agent.core.models import (
    REVIEW_CANCEL_CANDIDATE,
    REVIEW_HOLD,
    REVIEW_KEEP,
    LevelMap,
    MarketSnapshot,
    OrderRecord,
    OrderStatus,
    PositionReview,
)
from trading_agent.core.pnl import unrealized_pnl


def _age_minutes(opened_at: str | None) -> float | None:
    """Minutes since the order was opened, or None if the timestamp is unusable."""
    if not opened_at:
        return None
    try:
        parsed = datetime.fromisoformat(opened_at)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds() / 60.0


class PositionReviewAgent:
    name = "position_review_agent"

    def review(
        self,
        open_orders: list[OrderRecord],
        snapshots: dict[str, MarketSnapshot],
        level_maps: dict[str, LevelMap] | None,
        config: AppConfig,
    ) -> list[PositionReview]:
        level_maps = level_maps or {}
        min_hold_minutes = config.risk.min_hold_hours * 60
        reviews: list[PositionReview] = []
        for order in open_orders:
            snapshot = snapshots.get(order.symbol)
            mark = snapshot.last_price if snapshot is not None else None
            level_map = level_maps.get(order.symbol)
            regime = level_map.regime if level_map is not None else None

            age = _age_minutes(order.opened_at)
            min_hold_satisfied = age is not None and age >= min_hold_minutes

            to_tp = (
                round((order.take_profit_price / mark - 1) * 100, 4)
                if mark and order.take_profit_price
                else None
            )
            to_sl = (
                round((mark / order.stop_loss_price - 1) * 100, 4)
                if mark and order.stop_loss_price
                else None
            )
            _, unrealized_pct = unrealized_pnl(order, mark)

            action, reason = self._recommend(
                order=order,
                regime=regime,
                unrealized_pct=unrealized_pct,
                min_hold_satisfied=min_hold_satisfied,
            )
            reviews.append(
                PositionReview(
                    order_id=order.id,
                    symbol=order.symbol,
                    status=order.status.value,
                    recommended_action=action,
                    reason=reason,
                    age_minutes=round(age, 1) if age is not None else None,
                    min_hold_satisfied=min_hold_satisfied,
                    regime=regime,
                    current_price=mark,
                    unrealized_pnl_pct=unrealized_pct,
                    to_take_profit_pct=to_tp,
                    to_stop_loss_pct=to_sl,
                    stop_loss_price=order.stop_loss_price,
                    take_profit_price=order.take_profit_price,
                )
            )
        return reviews

    @staticmethod
    def _recommend(
        *,
        order: OrderRecord,
        regime: str | None,
        unrealized_pct: float | None,
        min_hold_satisfied: bool,
    ) -> tuple[str, str]:
        if order.status == OrderStatus.ENTRY_OPEN:
            # Resting (unfilled) bid: keep it working for the team's chosen horizon;
            # the TTL auto-expires it. Flag for cancel only if the bullish thesis the
            # bid was placed on has flipped to a confirmed downtrend.
            if regime == "downtrend":
                return (
                    REVIEW_CANCEL_CANDIDATE,
                    "regime flipped to downtrend; the bid's bullish thesis is gone; consider cancel",
                )
            return (REVIEW_KEEP, "resting bid still valid; keep working until filled or TTL")

        # Filled positions are no-touch: the exit ladder owns TP/SL/trailing, and
        # the only override is a critical news catalyst evaluated by the news
        # sentry. The recommendation is therefore always HOLD, annotated so the
        # sentry knows which symbols to watch first.
        if regime == "downtrend" and (unrealized_pct is not None and unrealized_pct < 0):
            return (
                REVIEW_HOLD,
                "downtrend and underwater; no-touch policy: the stop protects; news sentry should watch this symbol closely",
            )
        return (REVIEW_HOLD, "no-touch: exit ladder manages TP/SL; news sentry watches for critical catalysts")
