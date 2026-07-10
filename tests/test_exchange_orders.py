from __future__ import annotations

import json
import unittest
from urllib.request import Request

from trading_agent.core.config import Settings
from trading_agent.exchange import BinanceCredentials, BinanceSpotAdapter


class FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def read(self) -> bytes:
        return self.body


def make_adapter(body: str, requests: list[Request]) -> BinanceSpotAdapter:
    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        requests.append(request)
        return FakeResponse(body)

    adapter = BinanceSpotAdapter(
        base_url="https://testnet.binance.vision/api",
        settings=Settings(),
        urlopen=fake_urlopen,
    )
    # Pin the venue clock offset so signed calls don't emit a /v3/time sync
    # request first (these tests assert on the exact request sequence).
    adapter._time_offset_ms = 0
    return adapter


CREDS = BinanceCredentials(api_key="key", api_secret="secret")


class OrderStatusEndpointsTest(unittest.TestCase):
    def test_get_order_is_signed_get_on_v3_order(self) -> None:
        requests: list[Request] = []
        adapter = make_adapter(
            json.dumps({"orderId": 42, "status": "PARTIALLY_FILLED", "executedQty": "0.5"}),
            requests,
        )
        result = adapter.get_order(CREDS, "btcusdt", order_id=42)

        self.assertEqual(result["status"], "PARTIALLY_FILLED")
        self.assertEqual(requests[0].get_method(), "GET")
        self.assertIn("/v3/order?", requests[0].full_url)
        self.assertIn("symbol=BTCUSDT", requests[0].full_url)
        self.assertIn("orderId=42", requests[0].full_url)
        self.assertIn("signature=", requests[0].full_url)
        self.assertEqual(requests[0].get_header("X-mbx-apikey"), "key")

    def test_get_order_requires_an_identifier(self) -> None:
        adapter = make_adapter("{}", [])
        with self.assertRaises(ValueError):
            adapter.get_order(CREDS, "BTCUSDT")

    def test_get_order_by_client_order_id(self) -> None:
        requests: list[Request] = []
        adapter = make_adapter('{"status": "NEW"}', requests)
        adapter.get_order(CREDS, "BTCUSDT", client_order_id="taabc")
        self.assertIn("origClientOrderId=taabc", requests[0].full_url)

    def test_cancel_order_uses_delete(self) -> None:
        requests: list[Request] = []
        adapter = make_adapter('{"status": "CANCELED"}', requests)
        result = adapter.cancel_order(CREDS, "BTCUSDT", order_id=42)
        self.assertEqual(result["status"], "CANCELED")
        self.assertEqual(requests[0].get_method(), "DELETE")
        self.assertIn("/v3/order?", requests[0].full_url)

    def test_get_open_orders_signed(self) -> None:
        requests: list[Request] = []
        adapter = make_adapter("[]", requests)
        result = adapter.get_open_orders(CREDS, "BTCUSDT")
        self.assertEqual(result, [])
        self.assertIn("/v3/openOrders?", requests[0].full_url)
        self.assertIn("signature=", requests[0].full_url)

    def test_get_my_trades_filters_by_order(self) -> None:
        requests: list[Request] = []
        adapter = make_adapter(
            json.dumps([{"id": 1, "orderId": 42, "price": "100", "qty": "1", "commission": "0.1"}]),
            requests,
        )
        trades = adapter.get_my_trades(CREDS, "BTCUSDT", order_id=42)
        self.assertEqual(trades[0]["orderId"], 42)
        self.assertIn("/v3/myTrades?", requests[0].full_url)
        self.assertIn("orderId=42", requests[0].full_url)

    def test_get_klines_is_public(self) -> None:
        requests: list[Request] = []
        adapter = make_adapter(json.dumps([[1, "2", "3", "1", "2", "10", 2]]), requests)
        klines = adapter.get_klines("btcusdt", interval="15m", limit=2)
        self.assertEqual(len(klines), 1)
        self.assertIn("/v3/klines?", requests[0].full_url)
        self.assertIn("interval=15m", requests[0].full_url)
        self.assertNotIn("signature=", requests[0].full_url)
        self.assertIsNone(requests[0].get_header("X-mbx-apikey"))

    def test_book_ticker_parses_bid_ask(self) -> None:
        adapter = make_adapter(
            json.dumps(
                {"symbol": "BTCUSDT", "bidPrice": "99", "bidQty": "1", "askPrice": "101", "askQty": "2"}
            ),
            [],
        )
        ticker = adapter.book_ticker("BTCUSDT")
        self.assertEqual(ticker["bid_price"], "99")
        self.assertEqual(ticker["ask_price"], "101")


if __name__ == "__main__":
    unittest.main()
