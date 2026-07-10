"""Material-change cycle classifier: decide how much to spend on a cycle.

Pure (no I/O) so it is unit-testable. The expensive work is the deep-agent call
in ``consult_agents``; this decides whether it is worth making, given the cheap
deterministic context already built in ``prepare_context``. Exits are managed by
the fast bracket monitor between cycles, so a SKIP never leaves a position
unmanaged.
"""

from __future__ import annotations

from typing import Any

from trading_agent.core.config import CostConfig

FULL = "FULL"
REVIEW = "REVIEW"
SKIP = "SKIP"


def classify_cycle(
    *,
    baseline_intents: list[Any],
    open_views: list[dict[str, Any]],
    snapshots: dict[str, Any],
    last_marks: dict[str, float],
    minutes_since_review: float | None,
    is_first_cycle_of_day: bool,
    cost: CostConfig,
    has_entry_capacity: bool = True,
) -> tuple[str, str]:
    """Return (tier, reason). tier is FULL | REVIEW | SKIP."""
    if not cost.enabled:
        return FULL, "cost tiers disabled"
    # A baseline proposal only justifies FULL when it is a new entry with capacity
    # to take it (position cap + correlated budget): proposals for held symbols
    # and at-capacity entries are rejected by the gate anyway.
    open_symbols = {str(view.get("symbol")) for view in open_views}
    new_entry_signals = [
        intent for intent in baseline_intents if getattr(intent, "symbol", None) not in open_symbols
    ]
    if new_entry_signals and has_entry_capacity:
        return FULL, "deterministic baseline proposes a new entry"
    if cost.full_on_first_cycle_of_day and is_first_cycle_of_day:
        return FULL, "first cycle of the UTC day"

    # Open positions are exited mechanically (exit ladder + trailing stop), so
    # bracket proximity or a moving PnL warrants only the news-sentry REVIEW;
    # there is no bracket decision for the LLM to make. Without a quiet_model
    # these branches degrade to SKIP.
    sentry_tier = REVIEW if cost.quiet_model else SKIP
    for view in open_views:
        tp = _abs(view.get("to_take_profit_pct"))
        sl = _abs(view.get("to_stop_loss_pct"))
        pnl = _abs(view.get("unrealized_pnl_pct"))
        if tp is not None and tp <= cost.bracket_proximity_pct:
            return sentry_tier, f"{view.get('symbol')} within {cost.bracket_proximity_pct}% of take-profit"
        if sl is not None and sl <= cost.bracket_proximity_pct:
            return sentry_tier, f"{view.get('symbol')} within {cost.bracket_proximity_pct}% of stop"
        if pnl is not None and pnl >= cost.position_review_band_pct:
            return sentry_tier, f"{view.get('symbol')} unrealized PnL beyond +-{cost.position_review_band_pct}%"

    moved = _max_move_bps(snapshots, last_marks)
    if moved is not None and moved >= cost.material_move_bps:
        return FULL, f"price moved {moved:.0f}bps since last supervised cycle"

    if open_views and cost.quiet_model and (
        minutes_since_review is None or minutes_since_review >= cost.review_interval_minutes
    ):
        return REVIEW, "holding position(s); periodic review"

    return SKIP, "flat/quiet: no entry signal, no material move"


def _abs(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return None


def _max_move_bps(snapshots: dict[str, Any], last_marks: dict[str, float]) -> float | None:
    """Largest |% move| (in bps) of any symbol vs the last supervised marks."""
    best: float | None = None
    for symbol, snap in snapshots.items():
        prev = last_marks.get(symbol)
        last = getattr(snap, "last_price", None)
        if not prev or not last:
            continue
        move_bps = abs(last / prev - 1) * 10_000
        best = move_bps if best is None else max(best, move_bps)
    return best
