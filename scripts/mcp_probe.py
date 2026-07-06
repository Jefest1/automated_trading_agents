"""Empirical MCP reachability probe.

Tries each candidate remote MCP server with the real MultiServerMCPClient and
reports tool count / names / errors so we only enable endpoints that actually
connect and expose safe (non-execution) tools. Throwaway ops utility.
"""

from __future__ import annotations

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from trading_agent.utils.mcp_tools import is_disallowed_tool_name

CANDIDATES: list[tuple[str, str, str]] = [
    ("coingecko_http", "http", "https://mcp.api.coingecko.com/mcp"),
    ("fxmacrodata", "http", "https://fxmacrodata.com/mcp"),
    ("helium_http", "http", "https://heliumtrades.com/mcp"),
    ("helium_sse", "sse", "https://heliumtrades.com/sse"),
]


async def probe_one(name: str, transport: str, url: str, timeout: float = 25.0) -> None:
    conn = {name: {"transport": transport, "url": url}}
    try:
        client = MultiServerMCPClient(conn)
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - probe reports every failure mode
        print(f"[FAIL] {name:18} {transport:5} {url}\n        -> {type(exc).__name__}: {str(exc)[:200]}")
        return
    safe = [t for t in tools if not is_disallowed_tool_name(getattr(t, "name", ""))]
    blocked = len(tools) - len(safe)
    print(f"[OK]   {name:18} {transport:5} tools={len(tools)} safe={len(safe)} blocked={blocked}")
    for t in safe[:14]:
        desc = " ".join(str(getattr(t, "description", "")).split())[:110]
        print(f"        - {getattr(t, 'name', '?')}: {desc}")


async def main() -> None:
    for name, transport, url in CANDIDATES:
        await probe_one(name, transport, url)


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
