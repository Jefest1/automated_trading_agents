# Multi-Agent Trading Agent

A **virtual professional trading desk**: specialist subagents research, a
supervisor *deep agent* (the Portfolio Manager) makes the final
`BUY/SELL/WAIT/CLOSE/ADJUST` call, and a deterministic risk gate can veto or
resize anything before it reaches the exchange. Built on LangChain **deepagents**
+ **LangGraph** with a Claude Code-style REPL.

The desk trades a coherent **hourly swing** style - it decides once per hour off
1h/4h structure and holds winners for hours-to-days on a trailing exit ladder (it
is *not* an intraday scalper). Execution modes: **testnet** (testnet.binance.vision)
and **live** (api.binance.com / api.binance.us, multi-flag opt-in). Local paper
execution has been removed from the operational path.

See **[docs/architecture.md](docs/architecture.md)** for the full design with
**Mermaid diagrams** (system context, cycle pipeline, cadence tiers, agent
fan-out, exit ladder, async model) and [docs/mcp.md](docs/mcp.md) for wiring MCP
servers into the agents.

## How a cycle works

1. **Account review first** - every cycle starts from the account state and the
   **current date/time** (so news/search are anchored to today): balances, open
   positions, recent closed trades with per-trade realized PnL, the previous
   cycle, and a once-per-day **multi-timeframe brief** (compact 1M/1w/1d/4h/1h/15m
   indicators - the static higher-timeframe context studied at the day open, then
   relied on intra-day). Each open position is **marked to the live market**:
   current price, unrealized PnL (USD and %), distance to TP/SL, and age. The
   supervisor decides per symbol: BUY a new maker-pullback entry, or for an open
   position HOLD (WAIT) / SELL or CLOSE (exit via its order id) / ADJUST the
   bracket - banking profit or cutting losses on live PnL, not only waiting for
   the bracket. Before acting on an open order it resyncs against the exchange so
   its view matches live.
2. **Reconcile** - open testnet/live orders are synced against the exchange
   (live status via `GET /api/v3/order`, fills via `GET /api/v3/myTrades`);
   TP/SL brackets are monitored and exits submitted when touched.
3. **Research** - the supervisor dispatches market_research, news_research, and
   onchain_research subagents (in parallel). Each has live market data tools
   (`get_price`, `get_orderbook_ticker`, `get_klines` with computed
   EMA/SMA/RSI/MACD/ATR/Bollinger indicators), Binance Skills Hub read-only
   CLIs, web search/news search, and URL fetch (research agents visit the
   links news_research shares). The deterministic signal agents fall back to
   free keyless live sources before any placeholder: GDELT last-24h news and
   DefiLlama chain-TVL flow (`utils/free_feeds.py`).
4. **Decision** - the supervisor is the trader: after consulting **all six**
   subagents (research trio + strategy + risk_review + reporting) it emits a
   structured JSON decision per symbol: `BUY / SELL / WAIT / CLOSE / ADJUST`.
   Malformed or incomplete decisions degrade to WAIT.
5. **Deterministic gates** - before any order leaves the system:
   - pre-trade check: no duplicate positions, limit price within 2% of live
     market, and BUY limits must **rest at/below the bid** (a maker-pullback
     entry priced `min(llm_price, bid − risk.entry_atr_mult×ATR(1h))` so it never
     chases but a deeper bid at charted support is honored); a BUY above the bid
     by more than `risk.max_cross_spread_bps` is refused as chasing
   - RiskGovernor: symbol allowlist, confidence/edge thresholds, position caps,
     stale-data blocks, kill switch
   - exchange filters: price/quantity quantized to Binance tickSize/stepSize
     (prevents -1013 PRICE_FILTER rejections)
6. **Execution + per-trade PnL** - approved decisions execute as exchange
   limit orders. The order row is persisted as `PENDING_SUBMIT`
   **before** the exchange call, so a crash mid-submit leaves a client order id
   the reconciler can adopt or discard on restart (no orphaned positions).
   Realized PnL is computed **per round trip from actual fills including
   commissions** (BNB-denominated fees are converted at the current book price
   and flagged estimated), never from portfolio deltas.
7. **Sleep** - default decision interval is **60 minutes** (configurable in
   `config.json` or per-run with `/run --interval <seconds>`). Between cycles the
   fast bracket monitor manages open-position exits every ~60s, so a skipped cycle
   never leaves a position unmanaged.

**Cost tiers** (`config.json` `cost.*`): the cheap deterministic prep runs every
cycle, but the expensive LLM fan-out only fires when the cycle warrants it  - 
**FULL** (a new-entry signal, first cycle of the UTC day, a position near its
bracket, or a material price move), **REVIEW** (cheap manage-only model for a quiet
open position; requires `cost.quiet_model`), or **SKIP** (flat/quiet -> deterministic
WAIT, no LLM). See [docs/architecture.md §3](docs/architecture.md#3-cadence--cost-tiers).

The current implementation is intentionally conservative:

- Live trading requires an explicit multi-flag opt-in (see Execution modes).
- Only the deterministic execution layer can place or cancel orders - the
  supervisor decides, the risk gate can veto, deterministic code executes.
- Risk checks enforce symbol allowlists, stale-data blocks, open-position caps,
  third-position heuristic score, per-trade risk, kill switch, and full audit logging.
  Per-trade risk sizing prices in assumed slippage and an optional stop-gap
  buffer (a limit stop can fill worse than its price), not just the bare stop
  distance.
- A supervisor BUY must cite evidence ids that were actually gathered this cycle;
  fabricated `evidence_refs` are rejected (`require_evidence_refs`, default on)
  rather than silently re-scored, so the consultation trail stays auditable.
- The deterministic strategy's transfer function (agent weights, edge scale,
  confidence model) lives in `config.json` `strategy.*` so it can be tuned and
  backtested instead of being hard-coded.
- Placeholder evidence (emitted when a live source is unreachable) is excluded
  from strategy scoring and hard-rejected by the RiskGovernor - synthesized
  scores can never drive an order.
- Models are provider-agnostic (OpenAI, Anthropic, Google, Ollama, OpenRouter, local
  OpenAI-compatible servers) - swap with one env var.

This is software infrastructure, not financial advice.

## Quick Start

### Interactive REPL (recommended)

```powershell
uv run trading-agent --env-file .env repl
```

On startup the REPL reports your **exchange balances** and **open positions**, then
waits for commands:

```
/chat <message>                    talk to the team: ask prices, read charts, query
                                   orders/PnL, or instruct "close order ord_x" /
                                   "buy 0.001 BTC" (operator confirmation + risk
                                   gate apply). Plain text without a leading /
                                   also chats. Replies render as markdown panels.
/run [--symbols BTCUSDT ETHUSDT] [--interval 3600]
                                   start the agent loop (streams agent activity live;
                                   default interval = 60 min from config.json)
/once                              run a single supervised cycle and wait for it
/pause   /resume                   pause/resume at the next safe cycle boundary
/stop                              graceful stop (finishes the current cycle)
/status                            lifecycle state, mode, model, balances, positions
/balance                           exchange balances + open positions
/skills list                       installed Binance Skills Hub skills
/skills run query-token-info dynamic '{"symbol":"btc","currency":"USDT"}'
/search bitcoin etf flows          live web search
/news solana                       last-24h news search
/orders 20 [--sync]                recent orders; testnet/live status is refreshed
                                   from the exchange (per-trade PnL, fills)
/logs [50 | follow]                tail the agent log, or stream it live (Ctrl-C stops)
/kill on|off                       emergency kill switch (blocks all trading)
/report                            markdown operations report
/help   /exit                      (Ctrl-D also exits; Ctrl-C is ignored)
```

**Stopping a running loop:** type `/stop` (waits for the current cycle to finish),
or `/pause` to suspend without stopping. `/exit` stops the loop and quits the REPL.
`/kill on` instantly blocks all order submission without stopping research.

`.env` is the operator environment (testnet execution, LLM supervisor, live data).
`.env.test` is the hermetic placeholder environment used by the test suite.

### Classic one-shot CLI

```powershell
uv run trading-agent init
uv run trading-agent status
uv run trading-agent --env-file .env.test agent introduce
uv run trading-agent --env-file .env agent once --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT
uv run trading-agent signals --limit 20
uv run trading-agent orders --sync          # refresh live exchange status + per-trade PnL first
uv run trading-agent logs --lines 100       # last log lines
uv run trading-agent logs --follow          # live log stream (Ctrl-C to stop)
uv run trading-agent report --format markdown --output reports/latest.md
uv run trading-agent report --format json        # includes llm_cost (running token spend)
uv run trading-agent agent skills list
uv run trading-agent agent skills commands query-token-info
uv run trading-agent backtest --symbols BTCUSDT ETHUSDT --interval 1h --limit 500
uv run trading-agent backtest-decisions --interval 1h --window 48   # score the LLM's real decisions
uv run trading-agent --env-file .env mcp-check    # probe configured MCP servers
```

### Backtesting

`trading-agent backtest` replays historical public klines through the
deterministic pipeline (MarketDataAgent -> StrategyAgent -> RiskGovernor) with
the backtest fee/slippage model and reports realized PnL per symbol against two
benchmarks: buy-and-hold and WAIT-always. Use it to evaluate threshold changes
(`min_expected_edge_bps`, `min_confidence`, stops) on real history before
trusting them live.

`trading-agent backtest-decisions` measures the LLM supervisor instead of the
deterministic baseline. It replays the BUY decisions the supervisor actually
recorded (their limit price, quantity, and TP/SL bracket) through the same
fill/fee/slippage model against the real price path that *followed* each
decision, and reports realized PnL, win rate, per-symbol PnL, and a per-decision
buy-and-hold comparison. The model is not re-invoked (its news/onchain context
no longer exists) - the decision journal in the DB is the frozen context. This
is how the live trader's decisions get validated:

```powershell
uv run trading-agent backtest-decisions --interval 1h --window 48
```

### Model swapping

The supervisor model is provider-agnostic. Set in `.env`:

```
MODEL_PROVIDER=                            # optional; explicit override if set
                                           # openai | azure_openai | anthropic | google_genai | ollama | openrouter
MODEL_NAME=<override, else per-provider default>
MODEL_BASE_URL=<optional, for vLLM/LM Studio/OpenAI-compatible local servers>
```

Provider-specific env vars are auto-detected when `MODEL_PROVIDER` is blank.
Azure OpenAI wins when any Azure OpenAI env is configured:

```env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_PROJECT_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
```

Other providers use the matching key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`) or the generic `MODEL_API_KEY`.
Ollama needs no key - set `MODEL_PROVIDER=ollama` and run a local Ollama server.

### Live data

`TRADING_AGENT_LIVE_DATA=true` switches market snapshots and signal agents to
live sources: Binance Skills Hub CLI first, public Binance REST second,
deterministic simulation as the offline fallback. News sentiment additionally
uses free live web news search. Every evidence record carries its source label.

Binance Web3 skill CLIs are keyed by `(chainId, contractAddress)`, not spot
symbols; the runtime maps the majors to canonical pegged/wrapped contracts
(BTCB, Binance-Peg ETH, WBNB on BSC; wrapped SOL on Solana) before calling
`query-token-info dynamic`, `social-hype`, and `smart-money-inflow`. Symbols
without a mapping fall through to REST/simulation. Placeholder evidence is
flagged via its `-placeholder` source suffix and never trades (see above).

### Live research via MCP

Two keyless read-only MCP servers are enabled by default and their tools are
handed to the supervisor and research subagents each cycle: **crypto.com
market-data** (real-time prices/volume/rankings/trending) and **FXMacroData**
(FX/rates/COT/macro regime context). Each server is loaded in isolation with a 20s
timeout, so one bad endpoint can't stall a cycle or hide the others; a
`tool_allowlist` in `mcp_servers.json` trims the per-cycle schema footprint.
Verify with
`uv run trading-agent --env-file .env mcp-check`. Configure in
`.trading_agent/mcp_servers.json` - see [docs/mcp.md](docs/mcp.md).

By default the CLI stores state in `.trading_agent/agent.sqlite3` and configuration in `.trading_agent/config.json`.
Use `--env-file .env.test` to validate the placeholder environment through `trading_agent.core.config.Settings` without loading it into process globals. Real API keys should be provided through your shell or an operator-managed secret path, not committed.

### Tuning (`config.json`)

`decision_interval_minutes` sets the cycle cadence (default 60). Key `risk.*`
knobs:

- `min_confidence`, `min_expected_edge_bps` - decision thresholds.
- `max_open_positions`, `third_order_min_confidence`, `per_trade_risk_fraction`.
- `stop_loss_pct`, `take_profit_pct`, `stale_data_seconds`.
- `assumed_slippage_bps`, `stop_gap_buffer_pct` - per-trade risk sizing prices in
  assumed slippage and an optional stop-gap buffer (a limit stop can fill worse
  than its price), not just the bare stop distance.
- `atr_interval` (default `1h`) - the timeframe whose ATR sizes maker-pullback
  entries, stops, and the runner trail (the swing horizon).
- `stop_loss_pct` / `take_profit_pct` and `exits.*` - swing-width bracket: ~4%
  initial stop, TP tiers at +3% / +6%, runner on an ATR(1h) trailing stop.
- `entry_atr_mult`, `entry_min_offset_bps`, `entry_max_offset_pct` - maker-pullback
  entry sizing: applied limit = `min(llm_price, bid − entry_atr_mult×ATR(1h))`,
  clamped to the [min_bps, max_pct] band - never chases, honors a deeper support bid.
- `max_cross_spread_bps` - small tolerance for how far **above the bid** a BUY may
  sit (absorbs research->gate drift); beyond it the order is refused as chasing.
- `mark_refresh_seconds` - TTL for the live mark-to-market cache feeding
  unrealized PnL in `/orders`, `/balance`, and the loop heartbeat (default 300).
- `require_evidence_refs` - when true, a supervisor BUY whose `evidence_refs`
  resolve to no evidence gathered this cycle is rejected (no fabricated trails).

`strategy.*` exposes the deterministic baseline's transfer function so it can be
tuned and backtested instead of hard-coded: `agent_weights`, `edge_scale_bps`,
and the `confidence_*` coefficients.

### LLM cost tracking

Every cycle logs its real token spend and a running total (captured from the
provider's usage metadata across the supervisor and all subagents), priced with
`utils/token_cost.py` (GPT-5.4: $1.25 input / $0.125 cached / $10 output per 1M).
Routing the research subagents to a cheaper deployment (e.g. `gpt-5.4-nano`)
roughly halves the per-cycle cost:

```
token usage run_id=... cycle=3 input=... output=... reasoning=... cost_usd=1.24 | cumulative cycles=4 cost_usd=4.99
```

The running total is also in `report --format json` (`llm_cost`). Grep the log:

```powershell
uv run trading-agent logs --lines 4000 | Select-String "token usage" | Select-Object -Last 1
```

## Main Commands

```powershell
uv run python -m trading_agent.cli init
uv run python -m trading_agent.cli status
uv run python -m trading_agent.cli signals --limit 50
uv run python -m trading_agent.cli orders
uv run python -m trading_agent.cli risk config
uv run python -m trading_agent.cli kill-switch on
uv run python -m trading_agent.cli kill-switch off
uv run python -m trading_agent.cli report --format json
```

## Environment

Runtime environment validation lives in `src/trading_agent/core/config.py` as the Pydantic `Settings` class. The project does not use `python-dotenv` or `load_dotenv`.

Two environment files exist:

- **`.env`** - operator environment: Spot Testnet execution + orders enabled,
  LLM supervisor on, live data on, real (testnet) API keys. This is what you run with.
- **`.env.test`** - hermetic placeholder environment used by the unit-test suite:
  no secrets. Do not put keys here.

Key switches:

```env
TRADING_AGENT_EXECUTION_MODE=testnet      # testnet | live
TRADING_AGENT_ENABLE_TESTNET_ORDERS=true  # hard gate for real testnet submission
TRADING_AGENT_ENABLE_LIVE_ORDERS=false    # hard gate for live submission (see below)
TRADING_AGENT_ENABLE_LLM_SUPERVISOR=true  # deepagents supervisor on/off
TRADING_AGENT_LIVE_DATA=true              # live feeds vs deterministic simulation
MODEL_PROVIDER=                           # blank = auto-detect; or openai | azure_openai | anthropic | google_genai | ollama | openrouter
TRADING_AGENT_SUBAGENT_MODELS={"market_research":"azure_openai:gpt-5.4-nano"}
                                          # optional per-subagent model overrides
                                          # (default: subagents inherit the supervisor model;
                                          #  research scorers are routed to a cheap deployment)
```

### Execution modes

| Mode | Orders | Requirements |
|---|---|---|
| `testnet` | real orders on testnet.binance.vision | `TRADING_AGENT_ENABLE_TESTNET_ORDERS=true`, `BINANCE_VENUE=testnet`, keys |
| `live` | **real money** on binance.com/binance.us | ALL of: `TRADING_AGENT_ENABLE_LIVE_ORDERS=true`, `BINANCE_VENUE=binance.com` or `binance.us`, keys, and config.json `live.enabled=true` + `live.venue_confirmed=true`. Autonomous cycle orders additionally need `live.auto_orders_within_caps=true`; otherwise only operator-confirmed `/chat` orders execute. |

Market DATA always comes from the public production API regardless of mode  - 
testnet tickers diverge from the real market. Order EXECUTION uses the
mode-resolved base URL (`Settings.exchange_base_url()`).

Agent conversation memory persists in `<TRADING_AGENT_HOME>/checkpoints.sqlite3`
(LangGraph `AsyncSqliteSaver`), shared between supervised cycles and `/chat`.

If `TRADING_AGENT_ENABLE_LLM_SUPERVISOR=true`, the active provider's API key is required (`AZURE_OPENAI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `MODEL_API_KEY`; Ollama needs none). With the supervisor disabled, the deterministic pipeline still runs end-to-end.

Live logs stream to stderr and are also written to:

```text
<TRADING_AGENT_HOME>/logs/trading_agent.log    # agent cycles, gates, execution
<TRADING_AGENT_HOME>/logs/chat.log             # /chat transcript: operator messages,
                                               # supervisor replies, parsed decisions,
                                               # confirmed-order outcomes
```

Control logging with:

```env
TRADING_AGENT_LOG_LEVEL=INFO
TRADING_AGENT_LOG_TO_STDERR=true
TRADING_AGENT_LOG_TO_FILE=true
```

For verbose runtime tracing:

```powershell
uv run python -m trading_agent.cli --env-file .env --log-level DEBUG agent once --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT
```

Run the supervised watcher for a bounded time window:

```powershell
uv run python -m trading_agent.cli --env-file .env --log-level INFO agent run --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --duration-hours 6 --interval-seconds 900
uv run python -m trading_agent.cli --env-file .env --log-level INFO agent run --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --duration-days 2 --interval-seconds 900
```

The run loop logs market snapshots, each specialist evidence score, supervisor decisions, risk approvals/rejections, and submitted orders. Watch the logs live in another terminal:

```powershell
uv run trading-agent logs --follow
# or from inside the REPL: /logs follow
```

## Spot Testnet Orders

Exchange commands are separate from the agent loop. Signed order submission is hard-gated to `BINANCE_VENUE=testnet` and `BINANCE_API_BASE_URL=https://testnet.binance.vision/api`.

Validate a LIMIT order signature and filters without submitting:

```powershell
uv run python -m trading_agent.cli --env-file .env exchange testnet-limit-order --symbol BTCUSDT --side BUY --quantity 0.001 --price 90000.00
```

Submit to the Spot Testnet matching engine:

```powershell
uv run python -m trading_agent.cli --env-file .env exchange testnet-limit-order --symbol BTCUSDT --side BUY --quantity 0.001 --price 90000.00 --submit
```

Fetch a public testnet ticker:

```powershell
uv run python -m trading_agent.cli --env-file .env exchange ticker --symbol BTCUSDT
```

To let the supervised agent loop submit RiskGovernor-approved intents to Spot Testnet, your real `.env` must explicitly contain:

```env
TRADING_AGENT_MODE=testnet
TRADING_AGENT_EXECUTION_MODE=testnet
TRADING_AGENT_ENABLE_TESTNET_ORDERS=true
BINANCE_VENUE=testnet
BINANCE_API_BASE_URL=https://testnet.binance.vision/api
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

Then run a bounded watcher:

```powershell
uv run python -m trading_agent.cli --env-file .env --log-level INFO agent run --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --duration-hours 6 --interval-seconds 900
```

Agents still do not receive direct Binance order tools. The supervisor emits structured decisions; the deterministic gate approves or rejects; the deterministic runtime submits approved entries; the reconciler then tracks the order's LIVE status and fills from the exchange (the local DB is only a cache).

## Source Layout

- `src/trading_agent/core/`: settings, persisted app config, domain models, storage, risk, decision schema, exchange reconciliation (`exchange_sync.py`), per-trade PnL (`pnl.py`), and reporting.
- `src/trading_agent/agents/`: deterministic signal and strategy agents (quant baseline).
- `src/trading_agent/graph/`: LangGraph/Deep Agents runtime, split into `state / nodes / edges / cadence / compile / deep_agent / streaming / checkpointer / runtime` (see [docs/architecture.md](docs/architecture.md)).
- `src/trading_agent/prompts/`: versioned prompt registry (swing-style desk).
- `src/trading_agent/repl/`: operator console (`app / chat / renderer / events / lifecycle`).
- `src/trading_agent/utils/`: market data tools + indicators, feeds, web search, log tailing, ops tools, hosted MCP loading, read-only Binance skill wrappers, and the persistent-event-loop helper (`aioloop.py`).

## Research Dossier

See [docs/research_dossier.md](docs/research_dossier.md) for the architecture choices, source shortlist, safety controls, and staged implementation path.

## Binance Skills

Binance Skills Hub is installed under `.agents/skills` with the official `npx skills add https://github.com/binance/binance-skills-hub` flow. The runtime stages only read-only Binance research skills under `skills/binance-readonly` for Deep Agents.

Use `agent skills list` to inspect installed skills. Use `agent skills run <skill> <command> '<json>'` only for approved read-only Web3 research CLIs. Authenticated Binance CLI, wallet, payment, posting, transfer, order, and cancellation skills are installed for reference but are not exposed to agents as executable tools.

## Tests

```powershell
uv run python -m unittest discover -s tests
```
