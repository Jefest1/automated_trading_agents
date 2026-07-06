---
name: binance-readonly
description: Read-only Binance Skills Hub staging source for the trading-agent MVP. Use this as a safety reminder that only public-data Binance Web3 research skills are active; authenticated Binance CLI, wallet, payment, order, transfer, and cancellation skills are installed but not exposed to agents.
---

# Binance Read-Only Skill Source

This directory contains the curated Binance Skills Hub subset active for agents:

- crypto-market-rank
- meme-rush
- query-address-info
- query-token-audit
- query-token-info
- trading-signal
- binance-tokenized-securities-info

Only allowlisted public-data CLI commands may run through `run_binance_research_cli`.
Do not use authenticated Binance CLI, wallet, payment, order, transfer, posting, or cancellation commands from this source.
