from __future__ import annotations

import unittest
from unittest import mock

from trading_agent.agents.signals import NewsSentimentAgent, OnChainFlowAgent
from trading_agent.core.models import MarketSnapshot, utc_iso
from trading_agent.utils import free_feeds


def snapshot(symbol: str = "ETHUSDT") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        observed_at=utc_iso(),
        last_price=100.0,
        bid_price=99.5,
        ask_price=100.5,
        volume_24h=1000.0,
    )


class DefiLlamaFeedTest(unittest.TestCase):
    def setUp(self) -> None:
        free_feeds._cache.clear()

    def test_tvl_change_computed_from_last_two_points(self) -> None:
        series = [{"date": 1, "tvl": 90.0}, {"date": 2, "tvl": 100.0}, {"date": 3, "tvl": 105.0}]
        with mock.patch.object(free_feeds, "_fetch_json", return_value=series):
            flow = free_feeds.defillama_tvl_change("ETH")
        assert flow is not None
        self.assertEqual(flow["chain"], "Ethereum")
        self.assertAlmostEqual(flow["change_pct"], 0.05)

    def test_unknown_token_returns_none_without_fetch(self) -> None:
        with mock.patch.object(free_feeds, "_fetch_json") as fetch:
            self.assertIsNone(free_feeds.defillama_tvl_change("DOGE"))
        fetch.assert_not_called()

    def test_fetch_failure_returns_none(self) -> None:
        with mock.patch.object(free_feeds, "_fetch_json", return_value=None):
            self.assertIsNone(free_feeds.defillama_tvl_change("SOL"))


class GdeltFeedTest(unittest.TestCase):
    def setUp(self) -> None:
        free_feeds._cache.clear()

    def test_articles_parsed_to_headlines(self) -> None:
        payload = {
            "articles": [
                {"title": "Bitcoin rally continues", "url": "https://x/1", "seendate": "20260612"},
                {"title": "", "url": "https://x/skip"},
                {"title": "ETF approval boosts inflow", "url": "https://x/2", "seendate": "20260612"},
            ]
        }
        with mock.patch.object(free_feeds, "_fetch_json", return_value=payload):
            headlines = free_feeds.gdelt_headlines("bitcoin crypto")
        assert headlines is not None
        self.assertEqual(len(headlines), 2)
        self.assertEqual(headlines[0]["title"], "Bitcoin rally continues")

    def test_empty_response_returns_none(self) -> None:
        with mock.patch.object(free_feeds, "_fetch_json", return_value={"articles": []}):
            self.assertIsNone(free_feeds.gdelt_headlines("bitcoin crypto"))


class SignalAgentFallbackTest(unittest.TestCase):
    def test_onchain_agent_uses_derivatives_first(self) -> None:
        agent = OnChainFlowAgent(None, enable_defillama=True)
        deriv = {"symbol": "ETHUSDT", "funding_rate": 0.0005, "mark_price": 1700.0,
                 "open_interest": 1000.0, "oi_change_pct": 0.05}
        with mock.patch("trading_agent.agents.signals.binance_derivatives", return_value=deriv):
            record = agent.analyze("ETHUSDT", snapshot(), 1)
        self.assertEqual(record.source, "binance-derivatives")
        self.assertFalse(record.is_placeholder)
        self.assertGreater(record.score, 0.0)  # positive funding + rising OI = net-long

    def test_onchain_agent_uses_defillama_before_placeholder(self) -> None:
        agent = OnChainFlowAgent(None, enable_defillama=True)
        flow = {"chain": "Ethereum", "tvl_now_usd": 105.0, "tvl_prev_usd": 100.0, "change_pct": 0.05}
        with (
            mock.patch("trading_agent.agents.signals.binance_derivatives", return_value=None),
            mock.patch("trading_agent.agents.signals.defillama_tvl_change", return_value=flow),
        ):
            record = agent.analyze("ETHUSDT", snapshot(), 1)
        self.assertEqual(record.source, "defillama-tvl")
        self.assertFalse(record.is_placeholder)
        self.assertEqual(record.score, 1.0)  # +5% daily saturates the proxy

    def test_onchain_agent_offline_stays_placeholder(self) -> None:
        agent = OnChainFlowAgent(None, enable_defillama=False)
        with mock.patch("trading_agent.agents.signals.defillama_tvl_change") as feed:
            record = agent.analyze("ETHUSDT", snapshot(), 1)
        feed.assert_not_called()
        self.assertTrue(record.is_placeholder)

    def test_news_agent_uses_rss_first(self) -> None:
        agent = NewsSentimentAgent(None, enable_web_news=True)
        headlines = [
            {"title": "Ethereum rally and record inflow", "url": "https://x/1", "date": "20260612", "snippet": ""}
        ]
        with mock.patch("trading_agent.agents.signals.crypto_news_rss", return_value=headlines):
            record = agent.analyze("ETHUSDT", snapshot(), 1)
        self.assertEqual(record.source, "crypto-news-rss")
        self.assertFalse(record.is_placeholder)

    def test_news_agent_falls_back_to_gdelt(self) -> None:
        agent = NewsSentimentAgent(None, enable_web_news=True)
        headlines = [
            {"title": "Ethereum rally and record inflow", "url": "https://x/1", "date": "20260612", "snippet": ""}
        ]
        with (
            mock.patch("trading_agent.agents.signals.crypto_news_rss", return_value=None),
            mock.patch("trading_agent.agents.signals.run_web_news_search", return_value=[]),
            mock.patch("trading_agent.agents.signals.gdelt_headlines", return_value=headlines),
        ):
            record = agent.analyze("ETHUSDT", snapshot(), 1)
        self.assertEqual(record.source, "gdelt-news")
        self.assertFalse(record.is_placeholder)
        self.assertGreater(record.score, 0.0)


if __name__ == "__main__":
    unittest.main()
