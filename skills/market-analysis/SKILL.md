---
name: market-analysis
description: Analyze intraday Binance spot market structure into scored evidence. Use for OHLCV/indicators, trend, momentum, support/resistance, liquidity/spread, ATR, and macro regime — produce timestamped evidence, never execute trades.
---

# Market Analysis

Read the tape like a desk analyst: top-down, calibrated, and honest. One scored
evidence record per symbol. Your edge is describing what price is actually doing,
not constructing a bullish story.

## Method (top-down)

1. **Trend / structure (higher timeframe rules):** 4h then 1h. Above or below
   EMA20/50? Higher highs/lows or lower? Mark nearest support and resistance and
   the distance to each. Do not be bullish into overhead resistance with no room.
2. **Timing / momentum:** 15m RSI14 (overbought into resistance is a poor long),
   MACD histogram slope, EMA alignment, `momentum_bps = (last/close_N_ago - 1) * 10000`.
   Flag divergences (price up while momentum/volume fades = weak).
3. **Liquidity / execution:** live bid/ask and `spread_bps`; wide/unstable spread
   or a thin book lowers confidence and worsens fills. Record ATR(15m) — it sizes
   the stop and the maker-pullback entry offset.
4. **Regime:** fold in the macro read (USD strength, risk-on vs risk-off). Risk-off
   discounts longs; risk-on supports them.

## Output

- symbol, observed timestamp, source labels
- score -1..1 (bearish..bullish), confidence 0..1
- payload: trend_4h/1h, nearest_support, nearest_resistance, atr_15m, rsi_15m,
  momentum_bps, spread_bps, volume_24h, macro_regime
- stale / thin-liquidity / single-timeframe warnings (each cuts confidence)

## Scoring discipline

Aligned trend + healthy momentum + rising volume + tight spread + room to
resistance → toward +1. Momentum below ~15 bps round-trip cost, stretched into
resistance, or thinning volume → near 0. Broken structure or risk-off regime →
negative. Never inflate confidence to manufacture a trade.

Produce evidence only. Do not place, cancel, amend, or close orders, and do not
call any execution, wallet, or payment skill.
