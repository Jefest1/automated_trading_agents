from __future__ import annotations

import unittest

from trading_agent.graph.streaming import extract_agent_message, stream_deep_agent


class _MsgOnlyAgent:
    """Yields only a streamed supervisor model update (no root 'values' chunk),
    mirroring how deepagents+subgraphs actually streams."""

    def stream(self, payload, *, config, stream_mode, subgraphs):  # type: ignore[no-untyped-def]
        yield (("consult_agents:abc",), "updates", {"model": {"messages": [{"content": "decision text"}]}})


class _EmptyAgent:
    def stream(self, payload, *, config, stream_mode, subgraphs):  # type: ignore[no-untyped-def]
        return iter(())


class StreamingSmellTest(unittest.TestCase):
    def test_streamed_message_is_success_without_warning(self) -> None:
        emitted: list = []
        with self.assertNoLogs("trading_agent.streaming", level="WARNING"):
            state = stream_deep_agent(
                _MsgOnlyAgent(), {}, {}, lambda kind, agent, data: emitted.append((kind, agent))
            )
        self.assertIn("decision text", extract_agent_message(state))

    def test_truly_empty_stream_still_warns(self) -> None:
        with self.assertLogs("trading_agent.streaming", level="WARNING"):
            stream_deep_agent(_EmptyAgent(), {}, {}, lambda *a: None)


if __name__ == "__main__":
    unittest.main()
