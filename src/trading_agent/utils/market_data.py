"""Agent-facing market data tools backed by the Binance public REST API.

These tools are the only sanctioned source of current prices for agents:
prices must come from the exchange API, never be inferred from news or
recalled from model memory. Market data always uses the production public
endpoints (api.binance.com) — testnet tickers are thin and diverge from the
real market, which previously caused agents to quote prices that did not
match the charts.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.tools import tool

from trading_agent.core.logging import get_logger
from trading_agent.exchange import BinanceSpotAdapter
from trading_agent.utils import indicators
from trading_agent.utils.free_feeds import binance_derivatives

LOGGER = get_logger("market_data")

PUBLIC_MARKET_DATA_BASE_URL = "https://api.binance.com/api"

_VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}

# Test seam: swap the adapter without touching the tool signatures.
_adapter: BinanceSpotAdapter | None = None


def set_market_data_adapter(adapter: BinanceSpotAdapter | None) -> None:
    global _adapter
    _adapter = adapter


def _get_adapter() -> BinanceSpotAdapter:
    global _adapter
    if _adapter is None:
        _adapter = BinanceSpotAdapter(base_url=PUBLIC_MARKET_DATA_BASE_URL)
    return _adapter


def _error(tool_name: str, exc: Exception, **context: Any) -> str:
    LOGGER.warning("%s failed %s error=%s", tool_name, context, exc)
    return json.dumps({"ok": False, "error": str(exc), **context}, sort_keys=True)


@tool
def get_price(symbol: str) -> str:
    """Get the current live price of a spot symbol from the Binance public API.

    This is the authoritative current price — never infer prices from news,
    memory, or charts described by others. Example: get_price("BTCUSDT").
    Returns {"symbol", "price"}.
    """
    try:
        result = _get_adapter().ticker_price(symbol)
    except Exception as exc:
        return _error("get_price", exc, symbol=symbol)
    return json.dumps({"ok": True, "symbol": result["symbol"], "price": result["price"]}, sort_keys=True)


@tool
def get_orderbook_ticker(symbol: str) -> str:
    """Get the live best bid/ask (top of order book) for a spot symbol.

    Use before proposing any limit price: a BUY limit above the ask chases the
    market and will be rejected by the risk gate. Returns
    {"symbol", "bid_price", "bid_qty", "ask_price", "ask_qty"}.
    """
    try:
        result = _get_adapter().book_ticker(symbol)
    except Exception as exc:
        return _error("get_orderbook_ticker", exc, symbol=symbol)
    return json.dumps(
        {
            "ok": True,
            "symbol": result["symbol"],
            "bid_price": result["bid_price"],
            "bid_qty": result["bid_qty"],
            "ask_price": result["ask_price"],
            "ask_qty": result["ask_qty"],
        },
        sort_keys=True,
    )


@tool
def get_klines(symbol: str, interval: str = "15m", limit: int = 100) -> str:
    """Read the chart: OHLCV candles plus computed technical indicators.

    interval: one of 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
    (Binance spot notation; "1M" is one MONTH, "1m" is one minute).
    limit: number of candles (max 500 here).

    Returns JSON with:
    - "candles": the most recent candles as
      [open_time_ms, open, high, low, close, volume] (oldest first, last 30),
    - "indicators": latest EMA20/EMA50, SMA20, RSI14, MACD(12,26,9),
      ATR14, Bollinger(20,2) computed over the full requested window,
    - "summary": last close, window high/low, percent change over the window,
    - "sparkline": a unicode mini-chart of closes.

    Base every chart/trend claim on this data.
    """
    try:
        normalized_interval = interval if interval in _VALID_INTERVALS else "15m"
        capped_limit = max(20, min(int(limit), 500))
        raw = _get_adapter().get_klines(symbol, interval=normalized_interval, limit=capped_limit)
        candles = [
            [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
            for row in raw
        ]
    except Exception as exc:
        return _error("get_klines", exc, symbol=symbol, interval=interval)
    if not candles:
        return _error("get_klines", ValueError("no candles returned"), symbol=symbol, interval=interval)

    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    macd_line, signal_line, histogram = indicators.macd(closes)
    middle, upper, lower = indicators.bollinger(closes)
    first_close = closes[0]
    last_close = closes[-1]
    payload = {
        "ok": True,
        "symbol": symbol.upper(),
        "interval": normalized_interval,
        "candle_count": len(candles),
        "candles": candles[-30:],
        "indicators": {
            "ema_20": _round(indicators.latest(indicators.ema(closes, 20))),
            "ema_50": _round(indicators.latest(indicators.ema(closes, 50))),
            "sma_20": _round(indicators.latest(indicators.sma(closes, 20))),
            "rsi_14": _round(indicators.latest(indicators.rsi(closes, 14))),
            "macd": _round(indicators.latest(macd_line)),
            "macd_signal": _round(indicators.latest(signal_line)),
            "macd_histogram": _round(indicators.latest(histogram)),
            "atr_14": _round(indicators.latest(indicators.atr(highs, lows, closes, 14))),
            "bollinger_middle": _round(indicators.latest(middle)),
            "bollinger_upper": _round(indicators.latest(upper)),
            "bollinger_lower": _round(indicators.latest(lower)),
        },
        "summary": {
            "last_close": last_close,
            "window_high": max(highs),
            "window_low": min(lows),
            "change_pct": _round((last_close - first_close) / first_close * 100 if first_close else None),
        },
        "sparkline": indicators.ascii_sparkline(closes),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _round(value: float | None, digits: int = 8) -> float | None:
    return None if value is None else round(value, digits)


def current_prices(symbols: list[str]) -> dict[str, float]:
    """Best-effort live prices for a set of symbols (public API; failures skipped)."""
    prices: dict[str, float] = {}
    for symbol in dict.fromkeys(s.upper() for s in symbols):
        try:
            prices[symbol] = float(_get_adapter().ticker_price(symbol)["price"])
        except Exception as exc:
            LOGGER.warning("current_prices failed symbol=%s error=%s", symbol, exc)
    return prices


def cached_current_prices(symbols: list[str], store: Any, ttl: int = 300) -> dict[str, float]:
    """Live spot prices with a store-backed cache, refetched at most every ``ttl`` s.

    Marks are persisted in the ``last_marks`` setting ({symbol: {price, ts}}) so
    open-position unrealized PnL refreshes from Binance at most once per ttl
    window (default 5 min) across operator commands and the loop heartbeat.
    Network failures degrade to whatever cached marks exist (never raises).
    """
    wanted = list(dict.fromkeys(s.upper() for s in symbols))
    if not wanted:
        return {}
    marks = dict(store.get_setting("last_marks", {}) or {})
    now = time.time()
    out: dict[str, float] = {}
    stale: list[str] = []
    for symbol in wanted:
        mark = marks.get(symbol)
        if isinstance(mark, dict) and mark.get("price") and (now - float(mark.get("ts", 0))) < ttl:
            out[symbol] = float(mark["price"])
        else:
            stale.append(symbol)
    if stale:
        fresh = current_prices(stale)
        for symbol, price in fresh.items():
            marks[symbol] = {"price": price, "ts": now}
            out[symbol] = price
        if fresh:
            store.set_setting("last_marks", marks)
    return out


def atr_value(symbol: str, interval: str = "15m", period: int = 14, limit: int = 100) -> float | None:
    """Latest ATR for a symbol from public klines, or None on any failure.

    Used to size maker-pullback entries (limit = bid - k*ATR). Best-effort: never
    raises, so a market-data hiccup falls back to the configured minimum offset.
    """
    try:
        rows = _get_adapter().get_klines(symbol.upper(), interval=interval, limit=limit)
        highs = [float(row[2]) for row in rows]
        lows = [float(row[3]) for row in rows]
        closes = [float(row[4]) for row in rows]
        return indicators.latest(indicators.atr(highs, lows, closes, period=period))
    except Exception as exc:
        LOGGER.warning("atr_value failed symbol=%s error=%s", symbol, exc)
        return None


BRIEF_TIMEFRAMES = ("1M", "1w", "1d", "4h", "1h", "15m")


def _timeframe_summary(symbol: str, interval: str, limit: int = 60) -> dict[str, Any] | None:
    """Compact indicator snapshot for one symbol/timeframe (no raw candle dump)."""
    try:
        rows = _get_adapter().get_klines(symbol.upper(), interval=interval, limit=limit)
        highs = [float(row[2]) for row in rows]
        lows = [float(row[3]) for row in rows]
        closes = [float(row[4]) for row in rows]
        if not closes:
            return None
        macd_line, macd_signal, macd_hist = indicators.macd(closes)
        _, bb_upper, bb_lower = indicators.bollinger(closes)
        return {
            "candles": len(rows),
            "last_close": round(closes[-1], 8),
            "sma20": indicators.latest(indicators.sma(closes, 20)),
            "ema20": indicators.latest(indicators.ema(closes, 20)),
            "ema50": indicators.latest(indicators.ema(closes, 50)),
            "rsi14": indicators.latest(indicators.rsi(closes, 14)),
            "macd_hist": indicators.latest(macd_hist),
            "atr14": indicators.latest(indicators.atr(highs, lows, closes, 14)),
            "bb_upper": indicators.latest(bb_upper),
            "bb_lower": indicators.latest(bb_lower),
        }
    except Exception as exc:
        LOGGER.warning("timeframe summary failed symbol=%s interval=%s error=%s", symbol, interval, exc)
        return None


def multi_timeframe_brief(
    symbols: list[str], timeframes: tuple[str, ...] = BRIEF_TIMEFRAMES
) -> dict[str, Any]:
    """Compact multi-timeframe indicator brief per symbol (1M..15m) for the daily
    warm-up: the static higher-timeframe context the analyst studies once per day."""
    return {
        symbol.upper(): {tf: _timeframe_summary(symbol, tf) for tf in timeframes}
        for symbol in dict.fromkeys(s.upper() for s in symbols)
    }


# Per-timeframe candle depth for the levels engine. Higher TFs need fewer bars to
# cover years of structure; intraday TFs get more to map recent demand zones. 1M is
# capped by available history.
LEVEL_TIMEFRAME_LIMITS: tuple[tuple[str, int], ...] = (
    ("1M", 60),
    ("1w", 120),
    ("1d", 200),
    ("4h", 200),
    ("1h", 200),
)


def level_map_for(symbol: str, current_price: float | None = None):
    """Build a deterministic LevelMap (support/resistance zones + regime) for a symbol.

    Fetches monthly..1h public candles and runs the core.levels engine. Best-effort:
    returns None on any failure so a market-data hiccup degrades to the old shallow
    maker behaviour rather than blocking the cycle.
    """
    from trading_agent.core import levels as levels_engine

    try:
        candles_by_tf: dict[str, list] = {}
        for interval, limit in LEVEL_TIMEFRAME_LIMITS:
            rows = _get_adapter().get_klines(symbol.upper(), interval=interval, limit=limit)
            candles_by_tf[interval] = levels_engine.candles_from_klines(rows)
        price = current_price
        if price is None:
            daily = candles_by_tf.get("1d") or candles_by_tf.get("1h")
            if not daily:
                return None
            price = daily[-1].close
        return levels_engine.build_level_map(symbol.upper(), candles_by_tf, float(price))
    except Exception as exc:
        LOGGER.warning("level_map_for failed symbol=%s error=%s", symbol, exc)
        return None


@tool
def get_derivatives_positioning(symbol: str) -> str:
    """Funding rate + open interest for a major's perpetual (Binance futures, keyless).

    The real positioning / crowding read for BTC/ETH/SOL/BNB (the Web3 smart-money
    skills only cover meme tokens). Positive funding = net-long demand (longs pay
    shorts); an extreme reading is overcrowded/contrarian. Rising open interest =
    conviction building, falling = positions unwinding. Returns {funding_rate,
    funding_pct_8h, open_interest, oi_change_pct, mark_price}.
    """
    token = symbol.upper().removesuffix("USDT")
    try:
        deriv = binance_derivatives(token)
    except Exception as exc:
        return _error("get_derivatives_positioning", exc, symbol=symbol)
    if deriv is None:
        return json.dumps(
            {"ok": False, "error": "no derivatives data for symbol", "symbol": symbol},
            sort_keys=True,
        )
    return json.dumps(
        {
            "ok": True,
            "symbol": deriv["symbol"],
            "funding_rate": deriv["funding_rate"],
            "funding_pct_8h": _round(deriv["funding_rate"] * 100, 5),
            "open_interest": deriv["open_interest"],
            "oi_change_pct": _round(deriv["oi_change_pct"], 6) if deriv["oi_change_pct"] is not None else None,
            "mark_price": deriv["mark_price"],
            "positioning": "net-long" if deriv["funding_rate"] > 0 else "net-short" if deriv["funding_rate"] < 0 else "neutral",
        },
        sort_keys=True,
    )


MARKET_DATA_TOOLS = [get_price, get_orderbook_ticker, get_klines, get_derivatives_positioning]
