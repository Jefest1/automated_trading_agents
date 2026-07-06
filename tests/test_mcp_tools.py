from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from trading_agent.utils.mcp_tools import (
    MCPServerConfig,
    MCPToolLoader,
    build_multiserver_config,
    default_mcp_servers,
    is_disallowed_tool_name,
)


@dataclass(slots=True)
class FakeTool:
    name: str


class FakeClient:
    def __init__(self, connections: dict[str, dict[str, object]]) -> None:
        self.connections = connections

    async def get_tools(self) -> list[FakeTool]:
        return [FakeTool("read_market_snapshot"), FakeTool("place_order")]


class FailingClient:
    def __init__(self, connections: dict[str, dict[str, object]]) -> None:
        self.connections = connections

    async def get_tools(self) -> list[FakeTool]:
        raise RuntimeError("server unavailable")


class MCPToolLoaderTest(unittest.TestCase):
    def test_fake_mcp_tools_load_through_multiserver_client_and_filter_execution_tools(self) -> None:
        servers = [
            MCPServerConfig(
                name="research",
                transport="http",
                enabled=True,
                url="https://example.test/mcp",
            )
        ]
        result = MCPToolLoader(servers, client_cls=FakeClient).load_tools_sync()

        self.assertEqual([tool.name for tool in result.tools], ["read_market_snapshot"])
        self.assertEqual(result.blocked_tool_count, 1)
        self.assertIn("blocked", result.errors[0])

    def test_failed_mcp_load_is_recoverable(self) -> None:
        servers = [
            MCPServerConfig(
                name="research",
                transport="http",
                enabled=True,
                url="https://example.test/mcp",
            )
        ]
        result = MCPToolLoader(servers, client_cls=FailingClient).load_tools_sync()

        self.assertEqual(result.tools, [])
        self.assertIn("server unavailable", result.errors[0])

    def test_one_failing_server_does_not_blank_the_healthy_one(self) -> None:
        # Per-server isolation: a broken endpoint is reported but the healthy
        # server's tools still load (operational resilience).
        servers = [
            MCPServerConfig(name="good", transport="http", enabled=True, url="https://good.test/mcp"),
            MCPServerConfig(name="bad", transport="http", enabled=True, url="https://bad.test/mcp"),
        ]

        def client_factory(connections: dict[str, dict[str, object]]):
            return FailingClient(connections) if "bad" in connections else FakeClient(connections)

        result = MCPToolLoader(servers, client_cls=client_factory).load_tools_sync()  # type: ignore[arg-type]

        self.assertEqual([t.name for t in result.tools], ["read_market_snapshot"])
        self.assertEqual(result.server_count, 1)
        self.assertTrue(any("bad:" in e for e in result.errors))

    def test_slow_server_times_out_without_hanging(self) -> None:
        class HangingClient:
            def __init__(self, connections: dict[str, dict[str, object]]) -> None:
                self.connections = connections

            async def get_tools(self) -> list[FakeTool]:
                await asyncio.sleep(5)
                return [FakeTool("never")]

        servers = [MCPServerConfig(name="slow", transport="http", enabled=True, url="https://slow.test/mcp")]
        result = MCPToolLoader(servers, client_cls=HangingClient, timeout_seconds=0.1).load_tools_sync()  # type: ignore[arg-type]

        self.assertEqual(result.tools, [])
        self.assertTrue(any("timed out" in e for e in result.errors))

    def test_stdio_mcp_server_is_excluded_for_now(self) -> None:
        servers = [
            MCPServerConfig(
                name="local_stdio",
                transport="stdio",
                enabled=True,
                url="ignored",
            )
        ]

        self.assertEqual(build_multiserver_config(servers), {})

    def test_default_mcp_servers_are_hosted_and_disabled(self) -> None:
        servers = default_mcp_servers()

        self.assertTrue(servers)
        for server in servers:
            self.assertFalse(server.enabled)
            self.assertNotEqual(server.transport, "stdio")
            self.assertTrue(str(server.url).startswith("https://"))

    def test_execution_tool_names_are_blocked(self) -> None:
        self.assertTrue(is_disallowed_tool_name("place_order"))
        self.assertTrue(is_disallowed_tool_name("cancel_order"))
        self.assertFalse(is_disallowed_tool_name("query_docs"))


if __name__ == "__main__":
    unittest.main()
