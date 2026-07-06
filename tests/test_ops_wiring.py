from __future__ import annotations

import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.config import Settings
from trading_agent.core.logging import configure_logging, get_chat_logger, shutdown_logging
from trading_agent.graph.deep_agent import subagent_specs
from trading_agent.utils.ops_tools import build_ops_tools


class OpsToolsTest(unittest.TestCase):
    def test_ops_toolset_includes_order_and_decision_visibility(self) -> None:
        with TemporaryDirectory() as tmp:
            tools = build_ops_tools(Path(tmp) / "agent.sqlite3", Settings())
        names = {tool.name for tool in tools}
        self.assertEqual(
            names,
            {
                "list_open_orders",
                "get_order_status",
                "recent_trades_pnl",
                "recent_decisions",
                "sync_orders_from_exchange",
            },
        )

    def test_reporting_subagent_receives_injected_tools(self) -> None:
        # Regression: reporting had an empty toolset, so it could not actually
        # fetch PnL/open-position data it is prompted to summarize.
        with TemporaryDirectory() as tmp:
            ops = build_ops_tools(Path(tmp) / "agent.sqlite3", Settings())
        specs = {spec["name"]: spec for spec in subagent_specs(tools=ops, skills=[])}
        reporting_tools = {tool.name for tool in specs["reporting"]["tools"]}
        self.assertIn("list_open_orders", reporting_tools)
        self.assertIn("recent_decisions", reporting_tools)


class ChatLogTest(unittest.TestCase):
    def test_chat_logger_writes_to_dedicated_file_only(self) -> None:
        with TemporaryDirectory() as tmp:
            configure_logging(tmp, level="INFO", log_to_stderr=False, log_to_file=True)
            try:
                get_chat_logger().info("operator: buy 0.001 BTC")
                logging.getLogger("trading_agent.runtime").info("main log line")
                for handler in logging.getLogger("trading_agent_chat").handlers:
                    handler.flush()
                chat_log = (Path(tmp) / "logs" / "chat.log").read_text(encoding="utf-8")
                main_log = (Path(tmp) / "logs" / "trading_agent.log").read_text(encoding="utf-8")
            finally:
                shutdown_logging()
        self.assertIn("buy 0.001 BTC", chat_log)
        self.assertNotIn("buy 0.001 BTC", main_log)
        self.assertNotIn("main log line", chat_log)


if __name__ == "__main__":
    unittest.main()
