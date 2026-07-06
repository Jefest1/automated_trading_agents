from __future__ import annotations

import json
import unittest

from trading_agent.agents import signals
from trading_agent.agents.signals import NewsSentimentAgent, OnChainFlowAgent
from trading_agent.core.models import MarketSnapshot, utc_iso
from trading_agent.utils.binance_skills import MAJOR_TOKEN_CONTRACTS, BinanceSkillCommandResult
from trading_agent.utils.feeds import SimulatedMarketFeed
from trading_agent.utils.live_feed import BinanceLiveFeed


class FakeRegistry:
    """Records skill calls and replies with canned payloads per command."""

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def run_read_only_cli(self, skill_name: str, command: str, params_json: str, **_: object):
        params = json.loads(params_json)
        self.calls.append((skill_name, command, params))
        payload = self.payloads.get(command)
        if payload is None:
            return BinanceSkillCommandResult(skill_name, command, 1, "", "no canned payload")
        return BinanceSkillCommandResult(skill_name, command, 0, json.dumps(payload), "")


def dummy_snapshot(symbol: str = "BTCUSDT") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol, observed_at=utc_iso(), last_price=1.0, bid_price=1.0, ask_price=1.0, volume_24h=0.0
    )


class LiveFeedSkillParamsTest(unittest.TestCase):
    def setUp(self) -> None:
        signals._skill_cache.clear()

    def test_dynamic_uses_chain_and_contract(self) -> None:
        registry = FakeRegistry(
            {"dynamic": {"code": "000000", "data": {"price": "63000.5", "volume24h": "1234.5"}}}
        )
        feed = BinanceLiveFeed(registry, SimulatedMarketFeed())
        snapshot = feed.snapshot("BTCUSDT", 0)
        self.assertEqual(feed.last_source["BTCUSDT"], "binance-skills-hub")
        self.assertAlmostEqual(snapshot.last_price, 63000.5)
        skill, command, params = registry.calls[0]
        self.assertEqual((skill, command), ("query-token-info", "dynamic"))
        self.assertEqual(params["chainId"], MAJOR_TOKEN_CONTRACTS["BTC"]["chainId"])
        self.assertEqual(params["contractAddress"], MAJOR_TOKEN_CONTRACTS["BTC"]["contractAddress"])

    def test_unknown_symbol_falls_back_to_simulation(self) -> None:
        registry = FakeRegistry({})
        feed = BinanceLiveFeed(registry, SimulatedMarketFeed())
        snapshot = feed.snapshot("DOGEUSDT", 0)
        self.assertEqual(feed.last_source["DOGEUSDT"], "simulated")
        self.assertEqual(registry.calls, [])
        self.assertGreater(snapshot.last_price, 0)


class NewsSentimentSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        signals._skill_cache.clear()

    def test_scores_from_social_hype_leaderboard(self) -> None:
        contract = MAJOR_TOKEN_CONTRACTS["BTC"]
        registry = FakeRegistry(
            {
                "social-hype": {
                    "code": "000000",
                    "data": {
                        "leaderBoardList": [
                            {
                                "metaInfo": {
                                    "symbol": contract["wrappedSymbol"],
                                    "contractAddress": contract["contractAddress"],
                                },
                                "socialHypeInfo": {
                                    "sentiment": "Positive",
                                    "socialHype": 12345,
                                    "socialSummaryBrief": "ETF inflows",
                                },
                            }
                        ]
                    },
                }
            }
        )
        agent = NewsSentimentAgent(registry, enable_web_news=False)
        record = agent.analyze("BTCUSDT", dummy_snapshot(), 0)
        self.assertEqual(record.source, "binance-skills-hub")
        self.assertFalse(record.is_placeholder)
        self.assertGreater(record.score, 0)
        self.assertEqual(record.payload["sentiment"], "Positive")

    def test_falls_back_to_placeholder_when_token_absent(self) -> None:
        registry = FakeRegistry({"social-hype": {"code": "000000", "data": {"leaderBoardList": []}}})
        agent = NewsSentimentAgent(registry, enable_web_news=False)
        record = agent.analyze("BTCUSDT", dummy_snapshot(), 0)
        self.assertTrue(record.is_placeholder)


class OnChainFlowSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        signals._skill_cache.clear()

    def test_scores_from_smart_money_inflow(self) -> None:
        contract = MAJOR_TOKEN_CONTRACTS["SOL"]
        registry = FakeRegistry(
            {
                "smart-money-inflow": {
                    "code": "000000",
                    "data": [
                        {
                            "tokenName": "SOL",
                            "ca": contract["contractAddress"],
                            "inflow": 500000.0,
                            "traders": 42,
                        }
                    ],
                }
            }
        )
        agent = OnChainFlowAgent(registry)
        record = agent.analyze("SOLUSDT", dummy_snapshot("SOLUSDT"), 0)
        self.assertEqual(record.source, "binance-skills-hub")
        self.assertGreater(record.score, 0)
        self.assertEqual(record.payload["smart_money_traders"], 42)

    def test_leaderboard_call_is_cached_per_chain(self) -> None:
        contract = MAJOR_TOKEN_CONTRACTS["BTC"]
        registry = FakeRegistry(
            {
                "smart-money-inflow": {
                    "code": "000000",
                    "data": [{"ca": contract["contractAddress"], "inflow": -100000.0, "traders": 5}],
                }
            }
        )
        agent = OnChainFlowAgent(registry)
        agent.analyze("BTCUSDT", dummy_snapshot(), 0)
        agent.analyze("ETHUSDT", dummy_snapshot("ETHUSDT"), 0)  # same chain "56"
        inflow_calls = [call for call in registry.calls if call[1] == "smart-money-inflow"]
        self.assertEqual(len(inflow_calls), 1)


if __name__ == "__main__":
    unittest.main()
