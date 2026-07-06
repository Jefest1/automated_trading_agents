# Multi-Agent Trading Agent Research Dossier

> **2026-06-12 upgrade note:** the architecture below predates the
> supervisor-as-trader rework. Current design: [architecture.md](architecture.md)
> (decision pipeline, live exchange reconciliation, per-trade PnL from fills,
> async runtime + SQLite checkpointing, /chat) and [mcp.md](mcp.md). This
> dossier is kept for the original research and source shortlist.

## Decision Baseline

The MVP is a CLI-first, Binance-compatible spot trading system with paper trading first, then a small live-capital test only after the promotion gate passes. It starts with BTCUSDT, ETHUSDT, SOLUSDT, and BNBUSDT, and uses 15m-1h intraday decision cycles.

Locked operating defaults:

- Week one is paper trading only.
- Promotion requires positive net paper PnL after estimated fees/slippage and zero risk-control breaches.
- First live-capital budget is capped at 25-100 USD.
- Live trades may run automatically only within hard caps.
- Maximum open flat spot trades is 3; the third position requires at least 90% heuristic score.
- Initial order style is a plain spot limit entry with internally managed take-profit/stop-loss exits.
- Free-first data sources are preferred.
- OpenAI is the first model provider, but the code uses an OpenAI-compatible adapter boundary.

## Architecture Path

LangGraph remains the preferred future orchestration runtime because it is designed for long-running, stateful agent workflows with persistence, human-in-the-loop controls, streaming, and durable execution. The current repo implements the same boundaries in plain Python first so the control loop, storage schema, and safety rules can be tested before introducing orchestration dependencies.

AutoGen remains a research candidate for multi-agent collaboration and distributed agent workflows. NVIDIA NeMo Agent Toolkit, NemoClaw, OpenShell, and OpenClaw remain candidates for governed shell/tool execution when the CLI agent becomes always-on or remote-operated.

Sources:

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- AutoGen docs: https://microsoft.github.io/autogen/stable/
- NVIDIA NemoClaw: https://www.nvidia.com/en-us/ai/nemoclaw/
- NVIDIA NeMo Agent Toolkit: https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html

## Exchange And Data Sources

The exchange layer is adapter-first. Live trading must not assume Binance.com or Binance.US until the actual account venue and legal constraints are confirmed. The current code includes a safe Binance-compatible adapter boundary and only implements test-order plumbing as a future integration point.

Primary market data sources:

- Binance Spot REST market data: order book, trades, klines, ticker statistics.
- Binance WebSocket streams: klines, tickers, book updates.
- Binance Data Vision: historical data for backtesting.

External signal sources to research and integrate gradually:

- News: GDELT, RSS feeds, Binance announcements.
- On-chain: Binance Skills Hub, Etherscan, Solscan, DefiLlama.

Sources:

- Binance trading endpoints: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints
- Binance Spot Testnet: https://developers.binance.com/docs/binance-spot-api-docs/faqs/testnet
- Binance.US API: https://docs.binance.us/
- Binance market data: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- Binance WebSocket streams: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
- Binance Skills Hub: https://www.binance.com/en/skills
- GDELT: https://www.gdeltproject.org/
- Etherscan: https://docs.etherscan.io/
- Solscan: https://pro-api.solscan.io/pro-api-docs/v2.0/reference/v2-account-detail
- DefiLlama: https://docs.llama.fi/

## Implemented MVP Shape

The repository now has a working plain-Python MVP with these components:

- Market Data Agent: emits price/order-book evidence from a deterministic paper feed.
- News/Sentiment Agent: emits free-source style sentiment evidence through the same signal interface.
- On-Chain Flow Agent: emits deterministic on-chain flow evidence through the same signal interface.
- Strategy Agent: combines evidence into trade proposals.
- Risk Governor: enforces all locked safety defaults.
- Paper Execution Engine: submits plain spot limit entries and closes flat positions through internally managed take-profit/stop-loss exits.
- Logger/Evaluator: stores evidence, proposals, risk decisions, orders, events, and promotion reports in SQLite.
- CLI: status, paper run, signal inspection, open trades/orders, risk config, kill switch, and reports.

## Safety Rules

Only the execution component may submit or cancel orders. Every proposed order must pass the risk governor first.

Implemented checks:

- Kill switch blocks all new orders.
- Live mode is disabled unless explicitly configured and venue-confirmed.
- Symbol must be in the allowlist.
- Evidence must exist and be fresh.
- Open-position cap blocks flat spot trades beyond 3.
- The third open position requires at least 90% heuristic score.
- Proposal confidence must meet the configured minimum.
- Per-trade risk must fit the configured risk fraction.
- Live notional must fit the configured live-capital budget.
- All rejections are persisted as risk decisions and audit events.

## Next Implementation Steps

1. Add real Binance market data ingestion behind the existing feed interface.
2. Add GDELT/RSS/Binance-announcement ingestion behind the news agent.
3. Add Etherscan/Solscan/DefiLlama/Binance Skills Hub adapters behind the on-chain agent.
4. Add a LangGraph orchestration layer once the plain-Python control loop is stable.
5. Add Binance Spot Testnet validation for plain limit entry and plain spot exit behavior.
6. Add a read-only dashboard after CLI reports produce the right operational data.
