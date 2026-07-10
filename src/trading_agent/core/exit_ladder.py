"""Deterministic tiered exit ladder: scale-out take-profits + ratcheting stop.

The logic here is pure (no exchange, no DB) so it can be unit-tested in
isolation and reused by the exchange execution engine (``exchange_sync.py``). Each engine calls:

    1. ``update_trail(plan, mark, ...)``     - raise the high-water mark; trail the runner
    2. ``next_ladder_action(plan, mark, qty)`` - what to do this reconcile
    3. on a confirmed tier fill: ``apply_tier_fill(plan, tier, entry_price, cfg)``

It never mutates "filled" state itself except through ``apply_tier_fill`` so the
async exchange path can mark a tier filled only once the venue confirms the sell.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_agent.core.config import ExitConfig
from trading_agent.core.models import ExitLeg, ExitPlan, utc_iso


def build_exit_plan(
    entry_price: float,
    exit_config: ExitConfig,
    *,
    fallback_take_profit_pct: float,
    fallback_stop_loss_pct: float,
    stop_price: float | None = None,
) -> ExitPlan:
    """Construct the exit ladder for a freshly opened long at ``entry_price``.

    When ``exit_config.enabled`` is False this yields the legacy single-leg
    bracket (one TP at 100%, fixed stop) so behavior is unchanged on rollback.
    ``stop_price`` (absolute) overrides the computed stop for BOTH branches; a
    demand-zone bid must keep its stop just below the zone, not a fixed % off entry.
    """
    if not exit_config.enabled or not exit_config.take_profit_tiers:
        stop = stop_price if (stop_price and stop_price > 0) else entry_price * (1 - fallback_stop_loss_pct)
        return ExitPlan(
            legs=[
                ExitLeg(
                    tier=1,
                    target_price=round(entry_price * (1 + fallback_take_profit_pct), 8),
                    size_pct=1.0,
                )
            ],
            initial_stop_price=round(stop, 8),
            current_stop_price=round(stop, 8),
            high_water_price=entry_price,
            runner_size_pct=0.0,
            tiered=False,
        )

    legs = [
        ExitLeg(
            tier=index + 1,
            target_price=round(entry_price * (1 + tier["profit_pct"]), 8),
            size_pct=float(tier["size_pct"]),
        )
        for index, tier in enumerate(exit_config.take_profit_tiers)
    ]
    stop = stop_price if (stop_price and stop_price > 0) else entry_price * (1 - exit_config.initial_stop_loss_pct)
    return ExitPlan(
        legs=legs,
        initial_stop_price=round(stop, 8),
        current_stop_price=round(stop, 8),
        high_water_price=entry_price,
        runner_size_pct=exit_config.runner_size_pct,
        tiered=True,
    )


@dataclass(slots=True)
class LadderAction:
    """What the engine should do this reconcile. ``quantity`` is base asset."""

    kind: str  # "STOP_OUT" | "TAKE_TIER" | "NONE"
    reason: str = ""
    tier: int | None = None
    target_price: float | None = None
    quantity: float = 0.0


def tier_quantity(leg: ExitLeg, original_qty: float) -> float:
    return original_qty * leg.size_pct


def remaining_quantity(plan: ExitPlan, original_qty: float) -> float:
    """Base quantity not yet sold by any filled tier."""
    sold = sum(leg.filled_qty for leg in plan.legs if leg.filled)
    return max(0.0, original_qty - sold)


def update_trail(
    plan: ExitPlan, mark: float, exit_config: ExitConfig, *, atr: float | None = None
) -> bool:
    """Raise the high-water mark and, once the runner is active, ratchet the
    trailing stop UP toward it. Returns True if anything changed."""
    changed = False
    if mark > plan.high_water_price:
        plan.high_water_price = mark
        changed = True
    if plan.tiered and plan.runner_active and exit_config.trail_runner:
        if exit_config.trail_atr_mult is not None and atr:
            distance = exit_config.trail_atr_mult * atr
        else:
            distance = plan.high_water_price * exit_config.trail_pct
        candidate = round(plan.high_water_price - distance, 8)
        if candidate > plan.current_stop_price:
            plan.current_stop_price = candidate
            changed = True
    return changed


def next_ladder_action(plan: ExitPlan, mark: float, original_qty: float) -> LadderAction:
    """Decide the single action for this reconcile (stop first, then one tier)."""
    if remaining_quantity(plan, original_qty) <= 0:
        return LadderAction(kind="NONE")
    if mark <= plan.current_stop_price:
        return LadderAction(
            kind="STOP_OUT",
            reason=stop_reason(plan),
            quantity=remaining_quantity(plan, original_qty),
        )
    for leg in plan.legs:
        if not leg.filled and mark >= leg.target_price:
            # Legacy single-leg bracket keeps the plain "TAKE_PROFIT" reason.
            reason = f"TAKE_PROFIT_{leg.tier}" if plan.tiered else "TAKE_PROFIT"
            return LadderAction(
                kind="TAKE_TIER",
                reason=reason,
                tier=leg.tier,
                target_price=leg.target_price,
                quantity=tier_quantity(leg, original_qty),
            )
    return LadderAction(kind="NONE")


def apply_tier_fill(
    plan: ExitPlan,
    tier: int,
    entry_price: float,
    exit_config: ExitConfig,
    *,
    filled_qty: float | None = None,
    exit_order_exchange_id: str | None = None,
    exit_client_order_id: str | None = None,
) -> None:
    """Mark a tier filled and ratchet the stop UP per the breakeven/lock rules."""
    leg = _leg(plan, tier)
    if leg is None or leg.filled:
        return
    leg.filled = True
    leg.filled_qty = leg.filled_qty if filled_qty is None else filled_qty
    leg.filled_at = utc_iso()
    if exit_order_exchange_id is not None:
        leg.exit_order_exchange_id = exit_order_exchange_id
    if exit_client_order_id is not None:
        leg.exit_client_order_id = exit_client_order_id

    new_stop = plan.current_stop_price
    if exit_config.move_stop_to_breakeven_after_tier and (
        tier == exit_config.move_stop_to_breakeven_after_tier
    ):
        new_stop = max(new_stop, entry_price)
    if exit_config.lock_stop_to_prior_tier_after_tier and (
        tier >= exit_config.lock_stop_to_prior_tier_after_tier
    ):
        # Every tier at/after the lock threshold ratchets the stop to the tier
        # below it (TP2 fill -> stop at TP1; TP3 fill -> stop at TP2, +6% locked),
        # so a pyramid-out ladder keeps banking as it climbs.
        prior = _leg(plan, tier - 1)
        new_stop = max(new_stop, prior.target_price if prior is not None else entry_price)
    plan.current_stop_price = round(new_stop, 8)


def _leg(plan: ExitPlan, tier: int) -> ExitLeg | None:
    for leg in plan.legs:
        if leg.tier == tier:
            return leg
    return None


def stop_reason(plan: ExitPlan) -> str:
    if not plan.tiered:
        return "STOP_LOSS"
    if plan.runner_active:
        return "TRAIL_STOP"
    if plan.tiers_filled > 0:
        return "BREAKEVEN_STOP"
    return "STOP_LOSS"
