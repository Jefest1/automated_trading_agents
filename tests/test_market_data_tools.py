from __future__ import annotations

import json
import unittest

from trading_agent.utils import market_data


class FakeAdapter:
    def __init__(self) -> None:
        self.klines_payload = [
            [1718000000000 + i * 900_000, str(100 + i), str(101 + i), str(99 + i), str(100.5 + i), "10"]
            for i in range(60)
        ]

    def ticker_price(self, symbol: str) -> dict:
        return {"symbol": symbol.upper(), "price": "104999.50", "raw": {}}

    def book_ticker(self, symbol: str) -> dict:
        return {
            "symbol": symbol.upper(),
            "bid_price": "104999.0",
            "bid_qty": "2",
            "ask_price": "105001.0",
            "ask_qty": "3",
            "raw": {},
        }

    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> list:
        return self.klines_payload[:limit]


class FailingAdapter:
    def ticker_price(self, symbol: str) -> dict:
        raise RuntimeError("boom")


class MarketDataToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        market_data.set_market_data_adapter(FakeAdapter())  # type: ignore[arg-type]

    def tearDown(self) -> None:
        market_data.set_market_data_adapter(None)

    def test_get_price_returns_live_price(self) -> None:
        payload = json.loads(market_data.get_price.invoke({"symbol": "btcusdt"}))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["price"], "104999.50")

    def test_get_orderbook_ticker_returns_bid_ask(self) -> None:
        payload = json.loads(market_data.get_orderbook_ticker.invoke({"symbol": "BTCUSDT"}))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["bid_price"], "104999.0")
        self.assertEqual(payload["ask_price"], "105001.0")

    def test_get_klines_returns_candles_and_indicators(self) -> None:
        payload = json.loads(
            market_data.get_klines.invoke({"symbol": "BTCUSDT", "interval": "15m", "limit": 60})
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["candle_count"], 60)
        self.assertLessEqual(len(payload["candles"]), 30)
        self.assertIsNotNone(payload["indicators"]["ema_20"])
        self.assertIsNotNone(payload["indicators"]["rsi_14"])
        self.assertIsNotNone(payload["indicators"]["macd"])
        self.assertIsNotNone(payload["indicators"]["atr_14"])
        self.assertGreater(payload["summary"]["change_pct"], 0)
        self.assertTrue(payload["sparkline"])

    def test_invalid_interval_falls_back(self) -> None:
        payload = json.loads(
            market_data.get_klines.invoke({"symbol": "BTCUSDT", "interval": "15min", "limit": 60})
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["interval"], "15m")

    def test_errors_return_structured_failure(self) -> None:
        market_data.set_market_data_adapter(FailingAdapter())  # type: ignore[arg-type]
        payload = json.loads(market_data.get_price.invoke({"symbol": "BTCUSDT"}))
        self.assertFalse(payload["ok"])
        self.assertIn("boom", payload["error"])


if __name__ == "__main__":
    unittest.main()
