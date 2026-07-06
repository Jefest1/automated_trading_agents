---
name: reporting
description: Summarize trading-agent cycles into an audit-ready operator report — run/heartbeat, prompt versions, MCP/tool errors, decisions, risk outcomes, testnet/live positions, per-trade realized PnL, data-quality warnings, and promotion-gate status.
---

# Reporting

Summarize audit data from storage. Accuracy over optimism: never hide a loss, a
rejection, or a degraded data source. Keep research, decisions, risk outcomes,
execution events, and realized PnL clearly separated.

Include:

- latest run + heartbeat; prompt versions used
- per-symbol decisions (BUY/WAIT/CLOSE/SELL/ADJUST) with the one-line reason
- risk outcomes: RiskGovernor approvals/rejections with reasons, plus risk_review findings
- execution: orders submitted (ids, live exchange status), tiered exit fills / stop
  ratchets, reconciliation
- open and closed spot positions; **per-trade** realized PnL from actual fills
  (commissions included, flagged when estimated) — never collapse to a single
  portfolio number without the per-trade breakdown
- data-quality warnings (failing skills/MCP, empty news windows, repeated rejections)
- MCP/tool-call errors
- promotion-gate status: cumulative net PnL after fees **and after LLM cost**, plus
  remaining blockers

State numbers exactly as recorded; if a section has no data, write "none" rather
than dropping it. Reports must never trigger order placement, cancellation, or
execution.
