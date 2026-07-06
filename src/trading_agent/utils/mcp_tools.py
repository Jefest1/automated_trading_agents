from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from trading_agent.core.config import resolve_env_reference
from trading_agent.utils.aioloop import run_coro_blocking


DISALLOWED_TOOL_FRAGMENTS = (
    "submit_order",
    "place_order",
    "cancel_order",
    "amend_order",
    "execute_trade",
    "open_position",
    "close_position",
    "binance_order",
)


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    transport: str = "http"
    enabled: bool = False
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass(slots=True)
class MCPToolLoadResult:
    tools: list[Any]
    errors: list[str]
    server_count: int
    blocked_tool_count: int = 0
    # Tools dropped because they were not in the configured allowlist. MCP tool
    # SCHEMAS are injected into every agent that carries them (supervisor + the
    # research subagents), so an unused tool is pure context cost each cycle.
    filtered_tool_count: int = 0


def default_mcp_servers() -> list[MCPServerConfig]:
    return [
        MCPServerConfig(
            name="context7_docs",
            transport="http",
            url="https://mcp.context7.com/mcp",
            enabled=False,
            headers={"CONTEXT7_API_KEY": "env:CONTEXT7_API_KEY"},
            description="Hosted documentation MCP for current library/API docs.",
        ),
        MCPServerConfig(
            name="gitmcp_langgraph",
            transport="http",
            url="https://gitmcp.io/langchain-ai/langgraph",
            enabled=False,
            description="Hosted GitMCP context for the LangGraph repository.",
        ),
        MCPServerConfig(
            name="gitmcp_deepagents",
            transport="http",
            url="https://gitmcp.io/langchain-ai/deepagents",
            enabled=False,
            description="Hosted GitMCP context for the Deep Agents repository.",
        ),
        MCPServerConfig(
            name="gitmcp_binance_skills_hub",
            transport="http",
            url="https://gitmcp.io/binance/binance-skills-hub",
            enabled=False,
            description="Hosted GitMCP context for Binance Skills Hub research only.",
        ),
        MCPServerConfig(
            name="helium_news",
            transport="http",
            url="https://heliumtrades.com/mcp",
            enabled=False,
            description="DISABLED: Helium now returns HTTP 402 (subscription_required). News is "
            "served in-house via keyless web search (Jina -> DuckDuckGo -> GDELT) + social-hype.",
        ),
        MCPServerConfig(
            name="crypto_com_market_data",
            transport="http",
            url="https://mcp.crypto.com/market-data/mcp",
            enabled=False,
            description="Crypto.com remote MCP: real-time prices, market trends, volume, top "
            "rankings, trending tokens. Free, no API key. Read-only market data.",
        ),
        MCPServerConfig(
            name="fxmacrodata",
            transport="http",
            url="https://fxmacrodata.com/mcp",
            enabled=False,
            description="FXMacroData remote MCP: FX spot rates, central-bank policy rates, "
            "COT positioning, macro release calendar and briefings. Read-only.",
        ),
    ]


def mcp_config_path(home: str | Path) -> Path:
    return Path(home) / "mcp_servers.json"


def load_mcp_config(home: str | Path) -> list[MCPServerConfig]:
    path = mcp_config_path(home)
    if not path.exists():
        return default_mcp_servers()
    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = raw.get("servers", raw if isinstance(raw, list) else [])
    configs: list[MCPServerConfig] = []
    for item in servers:
        configs.append(
            MCPServerConfig(
                name=item["name"],
                transport=item["transport"],
                enabled=bool(item.get("enabled", False)),
                url=item.get("url"),
                env=dict(item.get("env", {})),
                headers=dict(item.get("headers", {})),
                description=item.get("description", ""),
            )
        )
    return configs


def load_mcp_tool_allowlist(home: str | Path) -> list[str] | None:
    """Optional ``tool_allowlist`` from mcp_servers.json (case-insensitive names).

    When present and non-empty, only MCP tools whose name is on the list are kept
    and handed to the agents; everything else is dropped before its schema is ever
    injected into a prompt. Absent/empty -> keep all loaded tools (legacy behavior).
    Hosted MCP servers (e.g. FXMacroData) expose dozens of niche tools — including
    image/``*_visual_artifact`` generators a text agent cannot consume — whose
    schemas otherwise cost ~20k tokens in every research agent every cycle.
    """
    path = mcp_config_path(home)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    allowlist = raw.get("tool_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        return None
    return [str(name) for name in allowlist]


def build_multiserver_config(
    servers: list[MCPServerConfig],
) -> dict[str, dict[str, Any]]:
    connections: dict[str, dict[str, Any]] = {}
    for server in servers:
        if not server.enabled:
            continue
        if server.transport == "stdio":
            continue
        connection: dict[str, Any] = {"transport": server.transport}
        if server.url:
            connection["url"] = server.url
        if server.env:
            connection["env"] = server.env
        headers = _resolve_headers(server.headers)
        if headers:
            connection["headers"] = headers
        connections[server.name] = connection
    return connections


def is_disallowed_tool_name(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in DISALLOWED_TOOL_FRAGMENTS)


class MCPToolLoader:
    def __init__(
        self,
        servers: list[MCPServerConfig],
        *,
        client_cls: type[MultiServerMCPClient] = MultiServerMCPClient,
        timeout_seconds: float = 20.0,
        tool_allowlist: list[str] | None = None,
    ) -> None:
        self.servers = servers
        self.client_cls = client_cls
        self.timeout_seconds = timeout_seconds
        # Case-insensitive set of tool names to keep; None/empty keeps all.
        self.tool_allowlist = (
            {name.lower() for name in tool_allowlist} if tool_allowlist else None
        )

    async def load_tools(self) -> MCPToolLoadResult:
        # Load each server in isolation with a timeout: a single slow or broken
        # MCP endpoint must not stall the cycle or blank out the tools from the
        # healthy servers (operational resilience over external dependencies).
        enabled = [
            server
            for server in self.servers
            if server.enabled and server.transport != "stdio"
        ]
        if not enabled:
            return MCPToolLoadResult(tools=[], errors=[], server_count=0)

        all_tools: list[Any] = []
        errors: list[str] = []
        loaded_servers = 0
        for server in enabled:
            connection = build_multiserver_config([server])
            if not connection:
                continue
            try:
                client = self.client_cls(connection)
                tools = await asyncio.wait_for(client.get_tools(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                errors.append(f"{server.name}: timed out after {self.timeout_seconds:g}s")
                continue
            except Exception as exc:  # pragma: no cover - exercised through fake clients
                errors.append(f"{server.name}: {exc}")
                continue
            loaded_servers += 1
            all_tools.extend(tools)

        safe_tools = []
        blocked = 0
        filtered = 0
        for tool in all_tools:
            name = str(getattr(tool, "name", ""))
            if is_disallowed_tool_name(name):
                blocked += 1
                continue
            # Drop tools outside the allowlist before their schema is ever sent.
            if self.tool_allowlist is not None and name.lower() not in self.tool_allowlist:
                filtered += 1
                continue
            safe_tools.append(_recoverable_tool(tool))
        if blocked:
            errors.insert(0, f"blocked {blocked} execution-capable MCP tool(s)")
        # NOTE: allowlist filtering is expected, normal behavior - it is reported
        # via filtered_tool_count (and logged), NOT added to errors, so it does not
        # inflate the cycle's error count every run.
        return MCPToolLoadResult(
            tools=safe_tools,
            errors=errors,
            server_count=loaded_servers,
            blocked_tool_count=blocked,
            filtered_tool_count=filtered,
        )

    def load_tools_sync(self) -> MCPToolLoadResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Persistent per-thread loop instead of asyncio.run: MCP HTTP sessions
            # are httpx clients, and closing a fresh loop each cycle orphans their
            # cleanup ("Event loop is closed"). See utils.aioloop.
            return run_coro_blocking(self.load_tools())
        raise RuntimeError(f"cannot synchronously load MCP tools while event loop {loop!r} is running")


def _resolve_headers(headers: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        env_value = resolve_env_reference(value)
        if env_value:
            resolved[key] = env_value
    return resolved


def _recoverable_tool(tool: Any) -> Any:
    """Return an MCP tool wrapper that reports remote failures as data.

    Hosted MCP tools are research inputs, not control-plane dependencies. A
    remote "ticker not found" or transient endpoint error should be visible to
    the model and logs, but it must not abort the entire supervisor graph.
    """
    if not (hasattr(tool, "invoke") or hasattr(tool, "ainvoke")):
        return tool

    name = str(getattr(tool, "name", "mcp_tool"))
    description = str(getattr(tool, "description", "") or f"Recoverable MCP tool {name}")
    args_schema = getattr(tool, "args_schema", None)

    def _error_payload(exc: Exception, kwargs: dict[str, Any]) -> str:
        return json.dumps(
            {
                "ok": False,
                "tool": name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "input": kwargs,
            },
            sort_keys=True,
            default=str,
        )

    def _safe_invoke(**kwargs: Any) -> Any:
        try:
            return tool.invoke(kwargs)
        except Exception as exc:
            return _error_payload(exc, kwargs)

    async def _safe_ainvoke(**kwargs: Any) -> Any:
        try:
            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(kwargs)
            return tool.invoke(kwargs)
        except Exception as exc:
            return _error_payload(exc, kwargs)

    return StructuredTool.from_function(
        func=_safe_invoke,
        coroutine=_safe_ainvoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )
