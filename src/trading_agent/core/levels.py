"""Deterministic support/resistance + demand-zone engine over multi-timeframe OHLCV.

This is the math half of the hybrid TA design: it computes *candidate* levels the
way a desk would draw them; fractal swing pivots, volume-by-price acceptance
(High-Volume Nodes = demand zones), prior month/week/day extremes, Fibonacci
retracement of the active swing, round numbers, and dynamic EMA support; then
clusters overlapping candidates into price *zones* (bands, not lines) scored by
confluence. A technical_analyst agent later confirms which zone to actually bid.

Everything here is pure (no I/O) so it is reproducible and unit-testable; the
fetch/assembly wrapper lives in utils.market_data. Levels are weighted by method
and timeframe (monthly/weekly structure dominates intraday noise), which is how a
professional reads top-down.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_agent.core.models import LevelMap, PriceLevel, SupportZone
from trading_agent.utils import indicators

# Higher timeframes carry more weight: a monthly level is structure, a 1h level is
# noise by comparison. Used as a multiplier on every level found on that timeframe.
TIMEFRAME_WEIGHT: dict[str, float] = {
    "1M": 3.0,
    "1w": 2.5,
    "1d": 2.0,
    "4h": 1.3,
    "1h": 1.0,
    "15m": 0.7,
}

# Method conviction: volume acceptance and prior-period extremes are the most
# reliable demand/supply markers; round numbers and fibs are softer confluence.
METHOD_WEIGHT: dict[str, float] = {
    "hvn": 1.5,
    "prior_low": 1.3,
    "prior_high": 1.3,
    "prior_close": 0.9,
    "swing_low": 1.2,
    "swing_high": 1.2,
    "fib": 0.8,
    "ema": 1.0,
    "round": 0.6,
}

DEFAULT_TIMEFRAMES = ("1M", "1w", "1d", "4h", "1h")


@dataclass(slots=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def candles_from_klines(rows: list[list]) -> list[Candle]:
    """Parse Binance kline rows ([open_time, o, h, l, c, v, ...]) into Candles."""
    candles: list[Candle] = []
    for row in rows:
        try:
            candles.append(
                Candle(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        except (IndexError, ValueError, TypeError):
            continue
    return candles


def fractal_pivots(candles: list[Candle], left: int = 2, right: int = 2) -> tuple[list[float], list[float]]:
    """Return (swing_highs, swing_lows) using N-bar fractals.

    A swing high is a bar whose high is >= the ``left`` bars before and > the
    ``right`` bars after it (strict on the right so a flat top is claimed once).
    Mirror logic for swing lows. Only fully-formed (closed) fractals are returned.
    """
    highs: list[float] = []
    lows: list[float] = []
    n = len(candles)
    for i in range(left, n - right):
        pivot = candles[i]
        window_l = candles[i - left : i]
        window_r = candles[i + 1 : i + 1 + right]
        if all(pivot.high >= c.high for c in window_l) and all(pivot.high > c.high for c in window_r):
            highs.append(pivot.high)
        if all(pivot.low <= c.low for c in window_l) and all(pivot.low < c.low for c in window_r):
            lows.append(pivot.low)
    return highs, lows


def volume_by_price(candles: list[Candle], bins: int = 24, top_quantile: float = 0.7) -> list[tuple[float, float]]:
    """High-Volume Nodes: price bins that absorbed the most traded volume.

    Volume is bucketed by each candle's typical price (hlc3). Bins whose volume is
    at/above the ``top_quantile`` of non-empty bins are returned as
    (price_center, volume); these are acceptance/demand zones where price spent
    real participation, the closest deterministic proxy to a drawn demand zone.
    """
    if len(candles) < 5:
        return []
    lows = min(c.low for c in candles)
    highs = max(c.high for c in candles)
    if highs <= lows:
        return []
    width = (highs - lows) / bins
    buckets = [0.0] * bins
    for c in candles:
        typical = (c.high + c.low + c.close) / 3.0
        idx = int((typical - lows) / width)
        idx = max(0, min(bins - 1, idx))
        buckets[idx] += c.volume
    nonzero = sorted(v for v in buckets if v > 0)
    if not nonzero:
        return []
    cutoff = nonzero[int(top_quantile * (len(nonzero) - 1))]
    nodes: list[tuple[float, float]] = []
    for idx, vol in enumerate(buckets):
        if vol >= cutoff and vol > 0:
            center = lows + (idx + 0.5) * width
            nodes.append((round(center, 8), vol))
    return nodes


def fib_retracement(swing_high: float, swing_low: float) -> dict[str, float]:
    """Standard retracement levels of the swing_low->swing_high impulse."""
    if swing_high <= swing_low:
        return {}
    span = swing_high - swing_low
    return {
        "0.382": round(swing_high - 0.382 * span, 8),
        "0.5": round(swing_high - 0.5 * span, 8),
        "0.618": round(swing_high - 0.618 * span, 8),
        "0.786": round(swing_high - 0.786 * span, 8),
    }


def round_number_step(price: float) -> float:
    """A sensible psychological-level step for the price magnitude.

    ~1% of price snapped to a 1/2/5 x 10^k grid, so BTC@64k -> 500, SOL@74 -> 0.5.
    """
    if price <= 0:
        return 0.0
    import math

    raw = price * 0.01
    power = math.floor(math.log10(raw))
    base = 10**power
    for mult in (1, 2, 5, 10):
        if base * mult >= raw:
            return float(base * mult)
    return float(base * 10)


def round_levels(price: float, count: int = 3) -> list[float]:
    """Nearest round-number levels above and below the price."""
    step = round_number_step(price)
    if step <= 0:
        return []
    nearest = round(price / step) * step
    levels = {round(nearest + i * step, 8) for i in range(-count, count + 1)}
    return sorted(levels)


def _prior_extremes(candles: list[Candle], timeframe: str) -> list[PriceLevel]:
    """Prior-period high/low/close from the last fully-closed candle on this TF.

    The most recent candle is forming; the one before it is the prior period
    (prior month/week/day), whose high/low/close are classic pivots desks watch.
    """
    if len(candles) < 2:
        return []
    prior = candles[-2]
    tw = TIMEFRAME_WEIGHT.get(timeframe, 1.0)
    return [
        PriceLevel(prior.high, "prior_high", timeframe, tw * METHOD_WEIGHT["prior_high"], f"prior_{timeframe}_high"),
        PriceLevel(prior.low, "prior_low", timeframe, tw * METHOD_WEIGHT["prior_low"], f"prior_{timeframe}_low"),
        PriceLevel(prior.close, "prior_close", timeframe, tw * METHOD_WEIGHT["prior_close"], f"prior_{timeframe}_close"),
    ]


def levels_for_timeframe(candles: list[Candle], timeframe: str) -> list[PriceLevel]:
    """All candidate levels detectable on one timeframe's candles."""
    if not candles:
        return []
    tw = TIMEFRAME_WEIGHT.get(timeframe, 1.0)
    levels: list[PriceLevel] = []

    highs, lows = fractal_pivots(candles)
    for value in highs:
        levels.append(PriceLevel(value, "swing_high", timeframe, tw * METHOD_WEIGHT["swing_high"], "swing_high"))
    for value in lows:
        levels.append(PriceLevel(value, "swing_low", timeframe, tw * METHOD_WEIGHT["swing_low"], "swing_low"))

    for center, _vol in volume_by_price(candles):
        levels.append(PriceLevel(center, "hvn", timeframe, tw * METHOD_WEIGHT["hvn"], "volume_node"))

    levels.extend(_prior_extremes(candles, timeframe))

    # Fib of the most recent dominant swing (highest high vs lowest low in window).
    hi = max(c.high for c in candles)
    lo = min(c.low for c in candles)
    for label, value in fib_retracement(hi, lo).items():
        levels.append(PriceLevel(value, "fib", timeframe, tw * METHOD_WEIGHT["fib"], f"fib_{label}"))

    # Dynamic EMA support/resistance (50/200) on the higher timeframes only.
    if timeframe in {"1M", "1w", "1d"} and len(candles) >= 50:
        closes = [c.close for c in candles]
        for period in (50, 200):
            if len(closes) >= period:
                ema_val = indicators.latest(indicators.ema(closes, period))
                if ema_val is not None:
                    levels.append(
                        PriceLevel(round(ema_val, 8), "ema", timeframe, tw * METHOD_WEIGHT["ema"], f"ema{period}")
                    )
    return levels


def cluster_levels(levels: list[PriceLevel], current_price: float, tolerance_pct: float = 0.006) -> list[SupportZone]:
    """Group nearby levels into confluence zones (bands).

    Levels are merged when within ``tolerance_pct`` of the running cluster mid.
    Zone strength = summed level weights (so 3 methods agreeing > 1 strong one);
    ``touches`` = number of contributing levels. side is relative to current price.
    """
    if not levels:
        return []
    ordered = sorted(levels, key=lambda lv: lv.price)
    clusters: list[list[PriceLevel]] = [[ordered[0]]]
    for lv in ordered[1:]:
        cluster = clusters[-1]
        cluster_mid = sum(item.price for item in cluster) / len(cluster)
        if cluster_mid > 0 and abs(lv.price - cluster_mid) / cluster_mid <= tolerance_pct:
            cluster.append(lv)
        else:
            clusters.append([lv])

    zones: list[SupportZone] = []
    for cluster in clusters:
        prices = [item.price for item in cluster]
        weight = sum(item.weight for item in cluster)
        mid = sum(prices) / len(prices)
        methods = sorted({item.kind for item in cluster})
        timeframes = sorted({item.timeframe for item in cluster}, key=lambda tf: -TIMEFRAME_WEIGHT.get(tf, 0))
        side = "support" if mid <= current_price else "resistance"
        distance_pct = (mid - current_price) / current_price if current_price > 0 else 0.0
        zones.append(
            SupportZone(
                low=round(min(prices), 8),
                high=round(max(prices), 8),
                mid=round(mid, 8),
                strength=round(weight, 4),
                side=side,
                methods=methods,
                timeframes=timeframes,
                distance_pct=round(distance_pct, 6),
                touches=len(cluster),
            )
        )
    return zones


def classify_regime(daily_closes: list[float], weekly_closes: list[float] | None = None) -> str:
    """Coarse trend regime: uptrend | range | downtrend.

    Uses the daily EMA20/EMA50 stack plus price location, confirmed by the weekly
    EMA20 slope when available. Designed to gate support-bidding: we bid in
    uptrend/range, not into a confirmed downtrend (knife-catching).
    """
    if len(daily_closes) < 50:
        return "range"
    ema20 = indicators.latest(indicators.ema(daily_closes, 20))
    ema50 = indicators.latest(indicators.ema(daily_closes, 50))
    price = daily_closes[-1]
    if ema20 is None or ema50 is None:
        return "range"

    weekly_up = weekly_down = False
    if weekly_closes and len(weekly_closes) >= 21:
        wema = indicators.ema(weekly_closes, 20)
        recent = [v for v in wema[-3:] if v is not None]
        if len(recent) >= 2:
            weekly_up = recent[-1] > recent[0]
            weekly_down = recent[-1] < recent[0]

    if price > ema20 > ema50 and not weekly_down:
        return "uptrend"
    if price < ema20 < ema50 and not weekly_up:
        return "downtrend"
    return "range"


def build_level_map(
    symbol: str,
    candles_by_tf: dict[str, list[Candle]],
    current_price: float,
    *,
    tolerance_pct: float = 0.006,
    max_zones_per_side: int = 6,
) -> LevelMap:
    """Assemble the full LevelMap from per-timeframe candles.

    Detects levels on every supplied timeframe, clusters them into confluence
    zones, classifies regime from daily/weekly closes, then splits zones into
    support (below price, nearest-first) and resistance (above, nearest-first).
    """
    all_levels: list[PriceLevel] = []
    for timeframe, candles in candles_by_tf.items():
        all_levels.extend(levels_for_timeframe(candles, timeframe))

    zones = cluster_levels(all_levels, current_price, tolerance_pct=tolerance_pct)

    daily = [c.close for c in candles_by_tf.get("1d", [])]
    weekly = [c.close for c in candles_by_tf.get("1w", [])]
    regime = classify_regime(daily, weekly)

    supports = sorted(
        (z for z in zones if z.side == "support" and z.high < current_price),
        key=lambda z: current_price - z.mid,
    )
    resistances = sorted(
        (z for z in zones if z.side == "resistance" and z.low > current_price),
        key=lambda z: z.mid - current_price,
    )
    return LevelMap(
        symbol=symbol,
        current_price=round(current_price, 8),
        regime=regime,
        support_zones=supports[:max_zones_per_side],
        resistance_zones=resistances[:max_zones_per_side],
    )
