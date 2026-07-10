"""Per-trade realized PnL computed from exchange fills.

PnL here is always per round trip (one entry order + its exit), never an
equity-wide aggregate. All amounts are expressed in the quote asset of the
traded symbol (e.g. USDT for BTCUSDT).

Commission handling:
- commission charged in the quote asset: subtracted directly;
- commission charged in the base asset: valued at that fill's price and
  subtracted (it reduces the base quantity actually received/delivered);
- commission charged in any other asset (e.g. BNB fee discount): converted
  with the caller-supplied `conversion_prices` (asset -> price in quote) and
  flagged `estimated=True` because the conversion price is approximate; with
  no conversion price the amount is recorded raw and also flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_agent.core.models import FillRecord, OrderRecord, OrderStatus


def unrealized_pnl(
    order: OrderRecord, current_price: float | None
) -> tuple[float | None, float | None]:
    """Mark an OPEN position to ``current_price``: (unrealized_usd, unrealized_pct).

    Returns (None, None) when not markable (no live price, not an open position,
    or no entry price). Shared by /orders, /balance, /report, and the loop's PnL
    heartbeat so the operator and agents see one consistent number.
    """
    if current_price is None or order.status != OrderStatus.POSITION_OPEN:
        return None, None
    entry = order.avg_fill_price or order.price
    quantity = order.executed_qty or order.quantity
    if not entry:
        return None, None
    # Only the HELD portion is still exposed: subtract quantity already sold by
    # filled take-profit tiers, so a scaled-out position is marked on what remains
    # (e.g. after TP1 banks 40%) rather than its full original size.
    plan = order.exit_plan
    if plan is not None:
        sold = sum(leg.filled_qty for leg in plan.legs if leg.filled)
        quantity = max(0.0, quantity - sold)
    return (
        round((current_price - entry) * quantity, 8),
        round((current_price / entry - 1) * 100, 4),
    )


@dataclass(slots=True)
class FillTotals:
    qty: float = 0.0
    quote_qty: float = 0.0
    commission_quote: float = 0.0
    unconverted_commissions: dict[str, float] = field(default_factory=dict)
    approx_converted_commissions: dict[str, float] = field(default_factory=dict)

    @property
    def avg_price(self) -> float | None:
        return self.quote_qty / self.qty if self.qty else None


@dataclass(slots=True)
class RoundTripPnl:
    realized_pnl: float
    entry: FillTotals
    exit: FillTotals
    commission_total_quote: float
    estimated: bool
    unconverted_commissions: dict[str, float]


def aggregate_fills(
    fills: list[FillRecord],
    *,
    base_asset: str,
    quote_asset: str,
    conversion_prices: dict[str, float] | None = None,
) -> FillTotals:
    totals = FillTotals()
    prices = {asset.upper(): price for asset, price in (conversion_prices or {}).items()}
    for fill in fills:
        totals.qty += fill.qty
        totals.quote_qty += fill.quote_qty
        asset = (fill.commission_asset or "").upper()
        if not fill.commission:
            continue
        if asset == quote_asset.upper():
            totals.commission_quote += fill.commission
        elif asset == base_asset.upper():
            totals.commission_quote += fill.commission * fill.price
        elif prices.get(asset):
            converted = fill.commission * prices[asset]
            totals.commission_quote += converted
            totals.approx_converted_commissions[asset] = (
                totals.approx_converted_commissions.get(asset, 0.0) + converted
            )
        else:
            totals.unconverted_commissions[asset] = (
                totals.unconverted_commissions.get(asset, 0.0) + fill.commission
            )
    return totals


def round_trip_pnl(
    entry_fills: list[FillRecord],
    exit_fills: list[FillRecord],
    *,
    base_asset: str,
    quote_asset: str,
    conversion_prices: dict[str, float] | None = None,
) -> RoundTripPnl:
    entry = aggregate_fills(
        entry_fills, base_asset=base_asset, quote_asset=quote_asset, conversion_prices=conversion_prices
    )
    exit_totals = aggregate_fills(
        exit_fills, base_asset=base_asset, quote_asset=quote_asset, conversion_prices=conversion_prices
    )
    # Match the entry COST BASIS to the quantity actually exited so a partial
    # scale-out (a TP tier filling while the runner stays open) realizes PnL only
    # on the sold portion. Booking the full entry cost against a partial exit
    # produced a large phantom loss. At full close exit_qty == entry_qty so
    # exit_fraction == 1.0 and the result is identical to before.
    exit_fraction = min(exit_totals.qty / entry.qty, 1.0) if entry.qty else 0.0
    matched_entry_cost = entry.quote_qty * exit_fraction
    matched_entry_commission = entry.commission_quote * exit_fraction
    commission_total = matched_entry_commission + exit_totals.commission_quote
    unconverted: dict[str, float] = dict(entry.unconverted_commissions)
    for asset, amount in exit_totals.unconverted_commissions.items():
        unconverted[asset] = unconverted.get(asset, 0.0) + amount
    approx = bool(entry.approx_converted_commissions or exit_totals.approx_converted_commissions)
    realized = exit_totals.quote_qty - matched_entry_cost - commission_total
    return RoundTripPnl(
        realized_pnl=round(realized, 8),
        entry=entry,
        exit=exit_totals,
        commission_total_quote=round(commission_total, 8),
        estimated=bool(unconverted) or approx,
        unconverted_commissions=unconverted,
    )


def realized_from_fills(
    order: OrderRecord,
    fills: list[FillRecord],
    *,
    conversion_prices: dict[str, float] | None = None,
) -> float:
    """Realized PnL for one order recomputed live FROM ITS FILLS (0.0 until any
    exit fills exist). Use this for display/aggregation instead of the cached
    ``order.realized_pnl`` field, which is only refreshed when a reconcile runs -
    so a value stored before a fix, or before the next cycle, would otherwise go
    stale. A partial scale-out yields the interim banked PnL on the sold portion.
    """
    entry_fills = [f for f in fills if not f.is_exit]
    exit_fills = [f for f in fills if f.is_exit]
    if not entry_fills or not exit_fills:
        return 0.0
    base_asset, quote_asset = split_symbol(order.symbol)
    result = round_trip_pnl(
        entry_fills,
        exit_fills,
        base_asset=base_asset,
        quote_asset=quote_asset,
        conversion_prices=conversion_prices,
    )
    return result.realized_pnl


def split_symbol(symbol: str) -> tuple[str, str]:
    """Best-effort base/quote split for common spot symbols (BTCUSDT -> BTC, USDT)."""
    upper = symbol.upper()
    for quote in ("USDT", "FDUSD", "USDC", "TUSD", "BUSD", "BTC", "ETH", "BNB", "EUR", "TRY"):
        if upper.endswith(quote) and len(upper) > len(quote):
            return upper[: -len(quote)], quote
    return upper, "USDT"
