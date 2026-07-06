---
name: evidence-scoring
description: Score trading-agent evidence consistently. Use when converting market, news, or on-chain observations into timestamped evidence with a calibrated score, confidence, source quality, recency/stale warnings, and an audit-ready rationale.
---

# Evidence Scoring

Bounded, calibrated scoring so the strategy layer can compare streams. The number
must be defensible from the cited data — not a vibe.

## Return

- evidence `kind`, source label or URL, observed timestamp (ISO-8601 UTC)
- `score` -1..1 (bearish..bullish), `confidence` 0..1
- short rationale (what specifically drove the score)
- warnings: stale, missing, conflicting, or low-quality-source

## Score (materiality, not direction alone)

- Strong, corroborated, fresh signal in one direction → toward ±1.
- Effect below round-trip cost (~25 bps), or ambiguous → near 0.
- "No signal" is a real finding: score 0.0, confidence ~0.5, with a note.

## Confidence (calibrate honestly)

- Two independent corroborating sources raise it; a single unverified source caps
  it at ~0.6.
- Anything older than 15 minutes is STALE → lower confidence and flag it.
- Wide spread, thin liquidity, single-timeframe reads, or proxy data (wrapped/pegged
  contracts standing in for the spot coin) all reduce confidence.
- Never raise confidence just to clear a downstream threshold.

Do not convert evidence into orders. TradeIntent creation belongs to strategy, and
execution belongs only to deterministic services.
