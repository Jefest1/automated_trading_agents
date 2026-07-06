---
name: risk-review
description: Pre-mortem and critique TradeIntent proposals before deterministic risk checks. Use to test evidence quality, freshness, edge-vs-cost, asymmetry, correlation/concentration, funding, and position limits. Advisory only — never approves or executes.
---

# Risk Review

Run a pre-mortem: assume the trade loses, name the most likely reason, then check
the proposal against the desk's rules. The deterministic `RiskGovernor` is the only
approval gate — your job is to catch the bad trade before it is placed. When in
doubt, raise the concern; a vetoed marginal trade is cheaper than a loss.

## Pre-mortem

For each non-WAIT proposal, state the single most likely way it goes wrong and the
concrete level/condition that would invalidate the thesis.

## Checklist

- Symbol is allowlisted; not already an open position (no adding to a position).
- Evidence: all three streams present, in schema, and fresh (none > 15 min STALE).
- Confidence is honest — consistent with the underlying evidence, not inflated to
  clear the bar.
- Edge vs cost: `expected_edge_bps >= 30` after ~25 bps round-trip, and the math
  matches the scores.
- Location & asymmetry: entry is not chasing into resistance or stretched momentum;
  a sensible stop gives reward:risk >= ~1.5.
- Correlation / concentration: would this stack another long onto already-open,
  highly-correlated majors (effectively one leveraged beta bet)?
- Position limits: within the configured open-position cap.
- Funding: free quote balance in the active execution environment covers the
  notional.
- History: a similar recent trade rejected or closed at a loss — does the rationale
  explain what changed? Kill-switch respected.

## Output

`{"intent_id": ..., "verdict": "looks_sound" | "concerns", "findings": ["<field +
threshold + why>"]}`. Be specific; never just "risky".

Do not override deterministic risk decisions and do not execute trades.
