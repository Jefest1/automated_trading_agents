from __future__ import annotations

import io
import time
import unittest
from types import SimpleNamespace

from rich.console import Console

from trading_agent.cli import _split_global_args
from trading_agent.repl.app import TradingAgentREPL
from trading_agent.repl.events import AgentEvent, EventBus
from trading_agent.repl.lifecycle import AgentLifecycleManager, AgentState
from trading_agent.repl.renderer import AgentRenderer


def cp1252_console() -> tuple[Console, io.BytesIO]:
    """A console writing to a cp1252 stream, like a legacy Windows terminal."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="replace", line_buffering=True)
    return Console(file=stream, legacy_windows=False, force_terminal=False, no_color=True), buffer


class SplitGlobalArgsTest(unittest.TestCase):
    def test_hoists_env_file_before_subcommand(self) -> None:
        self.assertEqual(
            _split_global_args(["--env-file", ".env.test"]),
            (["--env-file", ".env.test"], []),
        )

    def test_keeps_repl_args_after_subcommand(self) -> None:
        global_args, rest = _split_global_args(
            ["--env-file=.env", "--symbols", "BTCUSDT", "--log-level", "DEBUG"]
        )
        self.assertEqual(global_args, ["--env-file=.env", "--log-level", "DEBUG"])
        self.assertEqual(rest, ["--symbols", "BTCUSDT"])

    def test_empty_argv(self) -> None:
        self.assertEqual(_split_global_args([]), ([], []))


class RendererUnicodeTest(unittest.TestCase):
    def test_render_survives_unencodable_glyphs(self) -> None:
        console, _ = cp1252_console()
        renderer = AgentRenderer(console)
        # ◉ is the glyph that killed the UI pump on cp1252 consoles.
        renderer.render_event(AgentEvent(kind="token", agent="supervisor", data={"text": "scan ◉ done."}))
        renderer.flush_tokens()
        renderer.render_event(
            AgentEvent(kind="info", agent="supervisor", data={"message": "balances: ₿ 1.0"})
        )

    def test_render_event_kinds(self) -> None:
        console, buffer = cp1252_console()
        renderer = AgentRenderer(console)
        renderer.render_event(
            AgentEvent(
                kind="evidence",
                agent="market_data_agent",
                data={"symbol": "BTCUSDT", "kind": "price", "score": 0.5, "confidence": 0.8, "source": "live"},
            )
        )
        renderer.render_event(
            AgentEvent(kind="risk_decision", agent="risk_governor", data={"symbol": "BTCUSDT", "approved": True})
        )
        console.file.flush()
        text = buffer.getvalue().decode("cp1252")
        self.assertIn("BTCUSDT", text)
        self.assertIn("APPROVED", text)

    def test_tool_tokens_are_hidden_from_repl(self) -> None:
        console, buffer = cp1252_console()
        renderer = AgentRenderer(console)
        renderer.render_event(AgentEvent(kind="token", agent="tools", data={"text": "[[raw klines]]"}))
        renderer.render_event(
            AgentEvent(kind="token", agent="supervisor", data={"text": "Waiting for cleaner evidence."})
        )
        renderer.flush_tokens()
        console.file.flush()

        text = buffer.getvalue().decode("cp1252")
        self.assertNotIn("[[raw klines]]", text)
        self.assertIn("Waiting for cleaner evidence", text)


class UiPumpResilienceTest(unittest.TestCase):
    def test_render_safely_swallows_renderer_errors(self) -> None:
        class ExplodingRenderer:
            def render_event(self, event: AgentEvent) -> None:
                raise UnicodeEncodeError("charmap", "◉", 0, 1, "boom")

        stub = SimpleNamespace(renderer=ExplodingRenderer())
        event = AgentEvent(kind="token", agent="supervisor", data={"text": "◉"})
        # Must not raise: one bad event cannot kill the pump.
        TradingAgentREPL._render_safely(stub, event)


class LifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = EventBus()
        self.lifecycle = AgentLifecycleManager(self.bus)

    def test_start_run_stop_transitions(self) -> None:
        self.lifecycle.starting()
        self.lifecycle.running()
        self.assertEqual(self.lifecycle.state, AgentState.RUNNING)
        self.lifecycle.stop()
        self.assertTrue(self.lifecycle.stop_requested())
        self.assertFalse(self.lifecycle.wait_or_die(poll_seconds=0.01))
        self.lifecycle.stopped()
        self.assertEqual(self.lifecycle.state, AgentState.STOPPED)

    def test_pause_blocks_until_resume(self) -> None:
        self.lifecycle.starting()
        self.lifecycle.running()
        self.lifecycle.pause()
        self.assertEqual(self.lifecycle.state, AgentState.PAUSED)
        start = time.monotonic()
        self.lifecycle.resume()
        self.assertTrue(self.lifecycle.wait_or_die(poll_seconds=0.01))
        self.assertLess(time.monotonic() - start, 5.0)

    def test_skill_policy_gate(self) -> None:
        self.assertTrue(self.lifecycle.check_skill_allowed("query-token-info", "dynamic"))
        self.assertFalse(self.lifecycle.check_skill_allowed("query-token-info", "transfer"))
        self.assertFalse(self.lifecycle.check_skill_allowed("binance", "spot-order"))


if __name__ == "__main__":
    unittest.main()
