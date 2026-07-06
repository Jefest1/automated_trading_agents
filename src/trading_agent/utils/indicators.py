"""Pure-python technical indicators over OHLCV lists (no numpy/TA-Lib).

Each function takes plain float lists and returns a list aligned with the
input, padded with None while the indicator is warming up. `latest()` pulls
the most recent non-None value for compact agent-facing summaries.
"""

from __future__ import annotations


def latest(series: list[float | None]) -> float | None:
    for value in reversed(series):
        if value is not None:
            return value
    return None


def sma(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be > 0")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    multiplier = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    previous = seed
    for i in range(period, len(values)):
        previous = (values[i] - previous) * multiplier + previous
        out[i] = previous
    return out


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    """Wilder-smoothed Relative Strength Index."""
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def macd(
    values: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)
    macd_line: list[float | None] = [
        f - s if f is not None and s is not None else None
        for f, s in zip(fast_ema, slow_ema)
    ]
    macd_values = [v for v in macd_line if v is not None]
    signal_on_macd = ema(macd_values, signal)
    signal_line: list[float | None] = [None] * len(values)
    offset = len(values) - len(macd_values)
    for i, value in enumerate(signal_on_macd):
        signal_line[offset + i] = value
    histogram: list[float | None] = [
        m - s if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float | None]:
    """Wilder-smoothed Average True Range."""
    length = min(len(highs), len(lows), len(closes))
    out: list[float | None] = [None] * length
    if length <= period:
        return out
    true_ranges: list[float] = [highs[0] - lows[0]]
    for i in range(1, length):
        true_ranges.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    previous = sum(true_ranges[: period + 1]) / (period + 1)
    out[period] = previous
    for i in range(period + 1, length):
        previous = (previous * (period - 1) + true_ranges[i]) / period
        out[i] = previous
    return out


def bollinger(
    values: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (middle, upper, lower) bands."""
    middle = sma(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = middle[i]
        if mean is None:
            continue
        variance = sum((v - mean) ** 2 for v in window) / period
        std = variance**0.5
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return middle, upper, lower


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def ascii_sparkline(values: list[float], width: int = 40) -> str:
    """Compact unicode sparkline of a price series for terminal display."""
    if not values:
        return ""
    if len(values) > width:
        # Down-sample evenly so the sparkline spans the whole series.
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    low = min(values)
    high = max(values)
    if high == low:
        return _SPARK_CHARS[0] * len(values)
    scale = (len(_SPARK_CHARS) - 1) / (high - low)
    return "".join(_SPARK_CHARS[int((v - low) * scale)] for v in values)
