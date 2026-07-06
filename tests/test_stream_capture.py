from __future__ import annotations

import unittest
from typing import Any

from trading_agent.core.decision import parse_supervisor_decisions
from trading_agent.graph.streaming import (
    astream_deep_agent,
    extract_agent_message,
    stream_deep_agent,
)


class FakeStreamingAgent:
    """Mimics a deepagents graph streaming (namespace, mode, data) chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks

    def stream(self, payload: Any, *, config: Any, stream_mode: Any, subgraphs: bool) -> Any:
        assert "values" in stream_mode, "final state must be captured from values chunks"
        return iter(self.chunks)

    async def astream(self, payload: Any, *, config: Any, stream_mode: Any, subgraphs: bool) -> Any:
        assert "values" in stream_mode, "final state must be captured from values chunks"
        for chunk in self.chunks:
            yield chunk


class FakeBrokenCheckpointAgent(FakeStreamingAgent):
    def get_state(self, config: Any) -> Any:
        raise ValueError("No checkpointer set")

    async def aget_state(self, config: Any) -> Any:
        raise ValueError("No checkpointer set")


class StreamDeepAgentTest(unittest.TestCase):
    def test_returns_last_root_values_chunk(self) -> None:
        final = {"messages": ["m1", "m2"], "files": {"/skills/x": "y"}}
        chunks = [
            ((), "updates", {"model": {"messages": ["m1"]}}),
            (("market_research:123",), "updates", {"model": {"messages": ["sub"]}}),
            ((), "values", {"messages": ["m1"], "files": {}}),
            ((), "values", final),
        ]
        events: list[tuple[str, str, dict[str, Any]]] = []
        emit = lambda kind, agent, data: events.append((kind, agent, data))  # noqa: E731
        result = stream_deep_agent(FakeStreamingAgent(chunks), {}, {}, emit)
        self.assertEqual(result, final)
        # tool/node updates go to the file log, NOT the REPL event bus
        self.assertFalse(any(kind == "update" for kind, _, _ in events))

    def test_subgraph_values_do_not_override_root_state(self) -> None:
        root_final = {"messages": ["root"], "files": {}}
        chunks = [
            ((), "values", root_final),
            (("news_research:9",), "values", {"messages": ["subagent-internal"]}),
        ]
        result = stream_deep_agent(FakeStreamingAgent(chunks), {}, {}, lambda *args: None)
        self.assertEqual(result, root_final)

    def test_empty_stream_falls_back_without_crash(self) -> None:
        result = stream_deep_agent(FakeStreamingAgent([]), {}, {}, lambda *args: None)
        self.assertEqual(result, {})

    def test_uses_supervisor_model_update_when_values_chunk_is_missing(self) -> None:
        message = 'summary\n```json\n{"action": "WAIT", "symbol": "BTCUSDT", "rationale": "thin edge"}\n```'
        chunks = [
            (("consult_agents:abc",), "updates", {"model": {"messages": [{"content": message}]}}),
        ]
        result = stream_deep_agent(FakeBrokenCheckpointAgent(chunks), {}, {}, lambda *args: None)
        self.assertEqual(result["messages"][-1]["content"], message)

    def test_uses_supervisor_message_tokens_when_values_chunk_is_missing(self) -> None:
        chunks = [
            ((), "messages", ({"content": "part one "}, {})),
            ((), "messages", ({"content": "part two"}, {})),
        ]
        result = stream_deep_agent(FakeBrokenCheckpointAgent(chunks), {}, {}, lambda *args: None)
        self.assertEqual(result["messages"][-1]["content"], "part one part two")

    def test_tool_message_chunks_are_not_emitted_to_repl(self) -> None:
        chunks = [
            (
                (),
                "messages",
                ({"role": "tool", "content": "[[raw candle rows]]"}, {"langgraph_node": "tools"}),
            ),
        ]
        events: list[tuple[str, str, dict[str, Any]]] = []

        result = stream_deep_agent(
            FakeStreamingAgent(chunks),
            {},
            {},
            lambda kind, agent, data: events.append((kind, agent, data)),
        )

        self.assertEqual(events, [])
        self.assertEqual(result, {})

    def test_namespaced_subagent_message_chunks_are_emitted_to_repl(self) -> None:
        chunks = [
            (
                ("market_research:abc",),
                "messages",
                ({"content": "market sees momentum fading."}, {"langgraph_node": "model"}),
            ),
        ]
        events: list[tuple[str, str, dict[str, Any]]] = []

        result = stream_deep_agent(
            FakeStreamingAgent(chunks),
            {},
            {},
            lambda kind, agent, data: events.append((kind, agent, data)),
        )

        self.assertEqual(events, [("token", "market_research", {"text": "market sees momentum fading."})])
        self.assertEqual(result, {})

    def test_namespaced_model_updates_are_emitted_to_repl(self) -> None:
        chunks = [
            (
                ("risk_review:abc",),
                "updates",
                {"model": {"messages": [{"content": "risk review prefers wait."}]}},
            ),
        ]
        events: list[tuple[str, str, dict[str, Any]]] = []

        result = stream_deep_agent(
            FakeStreamingAgent(chunks),
            {},
            {},
            lambda kind, agent, data: events.append((kind, agent, data)),
        )

        self.assertEqual(events, [("token", "risk_review", {"text": "risk review prefers wait."})])
        self.assertEqual(result, {})

    def test_tool_updates_are_logged_not_emitted_to_repl(self) -> None:
        chunks = [
            (
                ("market_research:abc", "tools:def"),
                "updates",
                {"tools": {"messages": [{"role": "tool", "content": "[[raw klines]]"}]}},
            ),
        ]
        events: list[tuple[str, str, dict[str, Any]]] = []

        result = stream_deep_agent(
            FakeStreamingAgent(chunks),
            {},
            {},
            lambda kind, agent, data: events.append((kind, agent, data)),
        )

        self.assertEqual(events, [])
        self.assertEqual(result, {})

    def test_structural_tools_namespace_keeps_parent_agent_name_for_model_output(self) -> None:
        chunks = [
            (
                ("consult_agents:abc", "tools:def"),
                "updates",
                {"model": {"messages": [{"content": "consulted agents recommend WAIT."}]}},
            ),
        ]
        events: list[tuple[str, str, dict[str, Any]]] = []

        stream_deep_agent(
            FakeStreamingAgent(chunks),
            {},
            {},
            lambda kind, agent, data: events.append((kind, agent, data)),
        )

        self.assertEqual(events, [("token", "consult_agents", {"text": "consulted agents recommend WAIT."})])


class ExtractAgentMessageTest(unittest.TestCase):
    """The Responses API (gpt-5.x via Azure / OpenAI) returns message content as a
    LIST of typed blocks, not a string. extract_agent_message must flatten it to
    text so the fenced decision block survives for parse_supervisor_decisions."""

    def test_responses_api_list_content_preserves_decision_block(self) -> None:
        block_text = (
            "Analysis: thin edge in a corrective tape.\n"
            '```json\n{"action": "WAIT", "symbol": "BTCUSDT", "rationale": "no edge"}\n```'
        )
        response = {"messages": [{"content": [{"type": "text", "text": block_text}]}]}
        message = extract_agent_message(response)
        self.assertIn("```json", message)
        self.assertIn('"action"', message)
        decisions, errors = parse_supervisor_decisions(message)
        self.assertEqual(errors, [])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].symbol, "BTCUSDT")

    def test_plain_string_content_unchanged(self) -> None:
        response = {"messages": [{"content": "plain text reply"}]}
        self.assertEqual(extract_agent_message(response), "plain text reply")

    def test_multiple_text_blocks_are_joined(self) -> None:
        response = {
            "messages": [{"content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]
        }
        self.assertEqual(extract_agent_message(response), "part one\npart two")


class AsyncStreamDeepAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_async_stream_captures_root_values(self) -> None:
        final = {"messages": ["root-final"], "files": {}}
        chunks = [
            ((), "updates", {"model": {"messages": ["m1"]}}),
            ((), "values", final),
        ]
        result = await astream_deep_agent(FakeStreamingAgent(chunks), {}, {}, lambda *args: None)
        self.assertEqual(result, final)

    async def test_async_uses_supervisor_update_when_checkpoint_fallback_is_missing(self) -> None:
        message = 'final\n```json\n{"action": "WAIT", "symbol": "ETHUSDT", "rationale": "mixed"}\n```'
        chunks = [
            (("consult_agents:abc",), "updates", {"model": {"messages": [{"content": message}]}}),
        ]
        result = await astream_deep_agent(FakeBrokenCheckpointAgent(chunks), {}, {}, lambda *args: None)
        self.assertEqual(result["messages"][-1]["content"], message)


if __name__ == "__main__":
    unittest.main()
