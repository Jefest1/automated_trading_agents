from __future__ import annotations

import unittest

from trading_agent.agents.strategy import StrategyAgent
from trading_agent.core.config import AppConfig
from trading_agent.core.models import EvidenceRecord, MarketSnapshot, utc_iso


def _snapshot(symbol: str = "BTCUSDT") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        observed_at=utc_iso(),
        last_price=100.0,
        bid_price=99.9,
        ask_price=100.1,
        volume_24h=1000.0,
    )


def _evidence(symbol: str = "BTCUSDT", score: float = 0.2) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            agent="market_data_agent",
            source="test",
            symbol=symbol,
            kind="price_order_book",
            observed_at=utc_iso(),
            score=score,
            confidence=0.7,
            payload={},
        )
    ]


class StrategyTuningTest(unittest.TestCase):
    def test_edge_scale_is_config_driven(self) -> None:
        # score 0.2 * default edge_scale 120 = 24 bps < 30 min -> no proposal.
        base = AppConfig()
        self.assertEqual(
            StrategyAgent().propose({"BTCUSDT": _snapshot()}, _evidence(), base), []
        )

        # Raising the edge scale lifts the same evidence over the threshold.
        tuned = AppConfig()
        tuned.strategy.edge_scale_bps = 300.0
        proposals = StrategyAgent().propose({"BTCUSDT": _snapshot()}, _evidence(), tuned)
        self.assertEqual(len(proposals), 1)
        self.assertAlmostEqual(proposals[0].expected_edge_bps, 0.2 * 300.0, places=4)

    def test_agent_weights_are_config_driven(self) -> None:
        config = AppConfig()
        config.strategy.agent_weights = {"market_data_agent": 1.0}
        config.strategy.edge_scale_bps = 300.0
        proposals = StrategyAgent().propose(
            {"BTCUSDT": _snapshot()}, _evidence(score=0.5), config
        )
        self.assertEqual(len(proposals), 1)
        # Single agent at weight 1.0 -> combined score == its score.
        self.assertAlmostEqual(proposals[0].expected_edge_bps, 0.5 * 300.0, places=4)


if __name__ == "__main__":
    unittest.main()
