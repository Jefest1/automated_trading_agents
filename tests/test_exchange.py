from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request

from trading_agent.cli import main
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


class ExchangeAdapterTest(unittest.TestCase):
    def test_validate_limit_order_uses_test_endpoint_and_signature(self) -> None:
        requests: list[Request] = []

        def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
            requests.append(request)
            return FakeResponse("{}")

        adapter = BinanceSpotAdapter(
            base_url="https://testnet.binance.vision/api",
            settings=Settings(),
            urlopen=fake_urlopen,
        )
        result = adapter.validate_limit_order(
            BinanceCredentials(api_key="key", api_secret="secret"),
            "btcusdt",
            "BUY",
            "0.001",
            "90000.00",
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["submitted"])
        self.assertIn("/v3/order/test?", requests[0].full_url)
        self.assertIn("symbol=BTCUSDT", requests[0].full_url)
        self.assertIn("signature=", requests[0].full_url)
        self.assertEqual(requests[0].get_header("X-mbx-apikey"), "key")

    def test_submit_limit_order_uses_order_endpoint(self) -> None:
        requests: list[Request] = []

        def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
            requests.append(request)
            return FakeResponse('{"symbol":"BTCUSDT","orderId":123,"status":"NEW"}')

        adapter = BinanceSpotAdapter(
            base_url="https://testnet.binance.vision/api",
            settings=Settings(),
            urlopen=fake_urlopen,
        )
        result = adapter.submit_limit_order(
            BinanceCredentials(api_key="key", api_secret="secret"),
            "BTCUSDT",
            "BUY",
            "0.001",
            "90000.00",
        )

        self.assertTrue(result["submitted"])
        self.assertIn("/v3/order?", requests[0].full_url)
        self.assertNotIn("/v3/order/test?", requests[0].full_url)
        self.assertEqual(result["raw"]["orderId"], 123)

    def test_testnet_order_cli_rejects_non_testnet_venue(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"TRADING_AGENT_HOME={root}",
                        "BINANCE_VENUE=binance.com",
                        "BINANCE_API_BASE_URL=https://api.binance.com",
                        "BINANCE_API_KEY=key",
                        "BINANCE_API_SECRET=secret",
                        "TRADING_AGENT_LOG_TO_FILE=false",
                        "TRADING_AGENT_LOG_TO_STDERR=false",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "requires BINANCE_VENUE=testnet"):
                with redirect_stdout(StringIO()):
                    main(
                        [
                            "--env-file",
                            str(env_file),
                            "exchange",
                            "testnet-limit-order",
                            "--symbol",
                            "BTCUSDT",
                            "--side",
                            "BUY",
                            "--quantity",
                            "0.001",
                            "--price",
                            "90000.00",
                        ]
                    )

    def test_ticker_price_parses_public_response(self) -> None:
        def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
            return FakeResponse(json.dumps({"symbol": "BTCUSDT", "price": "100000.00"}))

        adapter = BinanceSpotAdapter(
            base_url="https://testnet.binance.vision/api",
            settings=Settings(),
            urlopen=fake_urlopen,
        )

        self.assertEqual(adapter.ticker_price("btcusdt")["price"], "100000.00")


if __name__ == "__main__":
    unittest.main()
