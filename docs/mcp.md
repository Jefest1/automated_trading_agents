# MCP servers

Agents use read-only research tools from configured MCP servers. The loader is
`src/trading_agent/utils/mcp_tools.py`; tools are loaded at the start of every
supervised cycle and handed to the supervisor **and** the research subagents.
Only remote transports are supported (`http` = streamable HTTP, `sse`,
`websocket`); `stdio`/local-process servers are ignored.

## Enabled by default

The shipped `.trading_agent/mcp_servers.json` enables two keyless, read-only
remote servers that the research subagents are **required** to use each cycle:

| Server | URL | What it adds |
|---|---|---|
| `crypto_com_market_data` | `https://mcp.crypto.com/market-data/mcp` | Real-time prices, market trends, volume, top rankings, trending tokens (`get_ticker`, `get_tickers`, `get_candlestick`, `get_instruments`, `get_book`). Price/volume cross-check for the research trio. |
| `fxmacrodata` | `https://fxmacrodata.com/mcp` | FX spot rates, central-bank policy rates, COT positioning, macro regime classifier/briefings - the macro context crypto reacts to. Mandatory macro read for `market_research`. |

A `tool_allowlist` in `mcp_servers.json` trims the handed-over tools to the ones
actually used (the full schema set is ~20k tokens/research-agent/cycle).
The GitMCP/Context7 documentation servers remain available but disabled.
CoinGecko (write-JavaScript `execute` only, never successfully called) and the
Helium news server were both removed as schema overhead for no data.

## Verify reachability

```powershell
uv run trading-agent --env-file .env mcp-check
```

Probes every enabled server independently and prints, per server, whether it is
reachable, how many tools loaded, the tool names, and any errors. Exit code is 0
only if every enabled server is reachable. Run it before a long session.

At runtime each cycle also logs `MCP tools loaded ... tools=N servers=M blocked=K`.

## Configuration

Edit `.trading_agent/mcp_servers.json`; set `"enabled": true/false` per server.
`config/mcp_servers.example.json` is the template. Each entry:

```json
{
  "name": "crypto_com_market_data",
  "transport": "http",
  "enabled": true,
  "url": "https://mcp.crypto.com/market-data/mcp",
  "headers": {},
  "env": {},
  "description": "..."
}
```

If the file is missing, `default_mcp_servers()` applies (all disabled).

## Resilience

Each enabled server is loaded **in isolation with a 20s timeout**, so one slow
or broken endpoint cannot stall a trading cycle or blank out the tools from the
healthy servers - its failure is logged and named, the rest still load. The
adapter is `langchain-mcp-adapters` (`MultiServerMCPClient`).

## Safety filter

Tool names matching order-execution fragments are blocked before they ever reach
an agent (`submit_order`, `place_order`, `cancel_order`, `amend_order`,
`execute_trade`, `open_position`, `close_position`, `binance_order`). Execution
flows exclusively through the decision JSON -> risk gate -> deterministic execution
path; MCP tools are research-only.
