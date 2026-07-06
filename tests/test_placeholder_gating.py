from __future__ import annotations

import unittest

from trading_agent.agents.strategy import StrategyAgent
from trading_agent.core.config import AppConfig
from trading_agent.core.models import EvidenceRecord, MarketSnapshot, Side, TradeProposal, utc_iso
from trading_agent.core.risk import RiskGovernor, RuntimeState


def record(agent: str, source: str, score: float, symbol: str = "BTCUSDT") -> EvidenceRecord:
    return EvidenceRecord(
        agent=agent,
        source=source,
        symbol=symbol,
        kind="test",
        observed_at=utc_iso(),
        score=score,
        confidence=0.8,
        payload={},
    )


def snapshot(symbol: str = "BTCUSDT") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        observed_at=utc_iso(),
        last_price=100.0,
        bid_price=99.99,
        ask_price=100.01,
        volume_24h=1000.0,
    )


class PlaceholderEvidenceGatingTest(unittest.TestCase):
    def test_evidence_record_placeholder_flag(self) -> None:
        self.assertTrue(record("a", "free-first-news-placeholder", 0.5).is_placeholder)
        self.assertFalse(record("a", "binance-skills-hub", 0.5).is_placeholder)

    def test_strategy_ignores_placeholder_only_symbols(self) -> None:
        strategy = StrategyAgent()
        evidence = [
            record("news_sentiment_agent", "free-first-news-placeholder", 0.9),
            record("onchain_flow_agent", "free-first-onchain-placeholder", 0.9),
        ]
        proposals = strategy.propose({"BTCUSDT": snapshot()}, evidence, AppConfig())
        self.assertEqual(proposals, [])

    def test_strategy_excludes_placeholder_scores_from_combination(self) -> None:
        strategy = StrategyAgent()
        config = AppConfig()
        live_market = record("market_data_agent", "binance-rest", 0.4)
        noisy_placeholder = record("news_sentiment_agent", "free-first-news-placeholder", 1.0)
        proposals = strategy.propose(
            {"BTCUSDT": snapshot()}, [live_market, noisy_placeholder], config
        )
        self.assertEqual(len(proposals), 1)
        # Combined score must equal the live market score alone (placeholder excluded).
        self.assertAlmostEqual(proposals[0].expected_edge_bps, 0.4 * 120.0, places=3)
        self.assertEqual(proposals[0].evidence_ids, [live_market.id])

    def test_source_quality_tiers(self) -> None:
        self.assertEqual(record("a", "binance-compatible-feed", 0.5).quality, 1.0)
        self.assertEqual(record("a", "binance-skills-hub", 0.5).quality, 1.0)
        self.assertLess(record("a", "defillama-tvl", 0.5).quality, 1.0)
        self.assertLess(record("a", "web-news", 0.5).quality, 1.0)
        self.assertEqual(record("a", "free-first-news-placeholder", 0.5).quality, 0.0)
        self.assertTrue(record("a", "defillama-tvl", 0.5).is_degraded)
        self.assertFalse(record("a", "binance-skills-hub", 0.5).is_degraded)

    def test_degraded_contradicting_source_is_down_weighted(self) -> None:
        # A bearish on-chain stream drags the combined score less when it comes
        # from a degraded source than from a primary one.
        strategy = StrategyAgent()
        config = AppConfig()
        market = record("market_data_agent", "binance-compatible-feed", 0.7)
        bearish_primary = record("onchain_flow_agent", "binance-skills-hub", -0.4)
        bearish_degraded = record("onchain_flow_agent", "defillama-tvl", -0.4)
        primary = strategy.propose({"BTCUSDT": snapshot()}, [market, bearish_primary], config)
        degraded = strategy.propose({"BTCUSDT": snapshot()}, [market, bearish_degraded], config)
        self.assertEqual(len(primary), 1)
        self.assertEqual(len(degraded), 1)
        # The degraded bearish drag is smaller, so the edge stays higher.
        self.assertGreater(degraded[0].expected_edge_bps, primary[0].expected_edge_bps)

    def test_risk_governor_rejects_placeholder_evidence(self) -> None:
        proposal = TradeProposal(
            symbol="BTCUSDT",
            side=Side.BUY,
            price=100.0,
            quantity=1.0,
            confidence=0.8,
            expected_edge_bps=30.0,
            risk_bps=100.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.015,
            rationale="test",
            evidence_ids=["ev"],
        )
        decision = RiskGovernor().evaluate(
            proposal,
            [record("news_sentiment_agent", "free-first-news-placeholder", 0.9)],
            RuntimeState(mode="testnet", open_position_count=0, kill_switch=False),
            AppConfig(),
        )
        self.assertFalse(decision.approved)
        self.assertIn("placeholder", " ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
