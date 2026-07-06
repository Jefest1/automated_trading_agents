from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.error import HTTPError, URLError

from trading_agent.core.config import Settings, load_settings

# Window the venue allows between our signed timestamp and its clock; 10s
# absorbs mild local clock drift (observed -1021 rejections at the default 5s).
_RECV_WINDOW_MS = 10_000
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (0.5, 1.5)
# Statuses safe to retry for reads: rate limits and transient server errors.
_TRANSIENT_STATUSES = {418, 429, 500, 502, 503, 504}


class BinanceRequestError(RuntimeError):
    """Failed Binance request; `status` is the HTTP code (None = network error)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _retryable(method: str, exc: BinanceRequestError) -> bool:
    if "-1021" in str(exc):
        return True  # stale timestamp; the retry re-signs with a fresh one
    if method == "POST":
        # Mutating call: retry only when the venue guarantees the request was
        # rejected before execution (rate limit). 5xx/timeouts are ambiguous —
        # the PENDING_SUBMIT reconciliation resolves those instead.
        return exc.status in {418, 429}
    if exc.status is None:
        return True  # network error on a read
    return exc.status in _TRANSIENT_STATUSES


@dataclass(slots=True)
class BinanceCredentials:
    api_key: str
    api_secret: str


class BinanceSpotAdapter:
    """Safe Binance-compatible adapter boundary.

    Live order submission is intentionally not wired into the CLI. This adapter
    supports signed test-order validation so exchange integration can be tested
    before any live trading path is enabled.
    """

    def __init__(
        self,
        base_url: str = "https://testnet.binance.vision/api",
        *,
        settings: Settings | None = None,
        urlopen: Any = urllib.request.urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.settings = settings or load_settings()
        self.urlopen = urlopen
        self._filters_cache: dict[str, dict[str, Decimal]] = {}

    @staticmethod
    def credentials_from_env(
        api_key_env: str = "BINANCE_API_KEY",
        api_secret_env: str = "BINANCE_API_SECRET",
        settings: Settings | None = None,
    ) -> BinanceCredentials:
        loaded_settings = settings or load_settings()
        api_key = loaded_settings.binance_api_key
        api_secret = loaded_settings.binance_api_secret
        if api_key is None or api_secret is None:
            raise RuntimeError(f"missing {api_key_env}/{api_secret_env}")
        return BinanceCredentials(api_key=api_key.get_secret_value(), api_secret=api_secret.get_secret_value())

    def test_limit_order(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        side: str,
        quantity: str | float,
        price: str | float,
    ) -> dict[str, Any]:
        return self.validate_limit_order(credentials, symbol, side, quantity, price)

    def validate_limit_order(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        side: str,
        quantity: str | float,
        price: str | float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return self._limit_order(
            credentials,
            symbol,
            side,
            quantity,
            price,
            endpoint="/v3/order/test",
            submitted=False,
            client_order_id=client_order_id,
        )

    def submit_limit_order(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        side: str,
        quantity: str | float,
        price: str | float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return self._limit_order(
            credentials,
            symbol,
            side,
            quantity,
            price,
            endpoint="/v3/order",
            submitted=True,
            client_order_id=client_order_id,
        )

    def ticker_price(self, symbol: str) -> dict[str, Any]:
        raw = self._public_request("GET", "/v3/ticker/price", {"symbol": symbol.upper()})
        return {"symbol": raw["symbol"], "price": raw["price"], "raw": raw}

    def book_ticker(self, symbol: str) -> dict[str, Any]:
        """Best bid/ask from the public order book ticker."""
        raw = self._public_request("GET", "/v3/ticker/bookTicker", {"symbol": symbol.upper()})
        return {
            "symbol": raw["symbol"],
            "bid_price": raw["bidPrice"],
            "bid_qty": raw["bidQty"],
            "ask_price": raw["askPrice"],
            "ask_qty": raw["askQty"],
            "raw": raw,
        }

    def get_klines(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 100,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list[Any]]:
        """Public OHLCV candles: [open_time, open, high, low, close, volume, close_time, ...].

        ``start_time``/``end_time`` are epoch milliseconds; with start_time the
        venue returns the forward window from that point (used by decision
        replay to score a decision against the price path that followed it).
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": max(1, min(int(limit), 1000)),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._public_request("GET", "/v3/klines", params)

    def get_order(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        *,
        order_id: int | str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Live order status straight from the exchange (never the local DB)."""
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("order_id or client_order_id is required")
        return self._signed_request(credentials, "GET", "/v3/order", params)

    def cancel_order(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        *,
        order_id: int | str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("order_id or client_order_id is required")
        return self._signed_request(credentials, "DELETE", "/v3/order", params)

    def get_open_orders(self, credentials: BinanceCredentials, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._signed_request(credentials, "GET", "/v3/openOrders", params)

    def get_my_trades(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        *,
        order_id: int | str | None = None,
        start_time: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Account fills (price/qty/commission per trade) for per-order PnL."""
        params: dict[str, Any] = {"symbol": symbol.upper(), "limit": max(1, min(int(limit), 1000))}
        if order_id is not None:
            params["orderId"] = order_id
        if start_time is not None:
            params["startTime"] = start_time
        return self._signed_request(credentials, "GET", "/v3/myTrades", params)

    def symbol_filters(self, symbol: str) -> dict[str, Decimal]:
        """Fetch (and cache) the PRICE_FILTER / LOT_SIZE constraints for a symbol.

        Orders whose price is not a multiple of tick_size, or whose quantity is
        not a multiple of step_size, are rejected by Binance with -1013.
        """
        normalized_symbol = symbol.upper()
        cached = self._filters_cache.get(normalized_symbol)
        if cached is not None:
            return cached
        raw = self._public_request("GET", "/v3/exchangeInfo", {"symbol": normalized_symbol})
        filters: dict[str, Decimal] = {}
        for entry in raw.get("symbols", [{}])[0].get("filters", []):
            if entry.get("filterType") == "PRICE_FILTER":
                filters["tick_size"] = Decimal(entry["tickSize"])
            elif entry.get("filterType") == "LOT_SIZE":
                filters["step_size"] = Decimal(entry["stepSize"])
                filters["min_qty"] = Decimal(entry["minQty"])
            elif entry.get("filterType") in {"NOTIONAL", "MIN_NOTIONAL"}:
                filters["min_notional"] = Decimal(entry.get("minNotional", "0"))
        self._filters_cache[normalized_symbol] = filters
        return filters

    def quantize_order(self, symbol: str, quantity: str | float, price: str | float) -> tuple[str, str]:
        """Round price down to tick_size and quantity down to step_size."""
        filters = self.symbol_filters(symbol)
        quantized_price = _quantize_down(Decimal(str(price)), filters.get("tick_size"))
        quantized_quantity = _quantize_down(Decimal(str(quantity)), filters.get("step_size"))
        min_qty = filters.get("min_qty")
        if min_qty is not None and quantized_quantity < min_qty:
            raise ValueError(
                f"quantity {quantized_quantity} below LOT_SIZE minQty {min_qty} for {symbol.upper()}"
            )
        min_notional = filters.get("min_notional")
        if min_notional is not None and quantized_price * quantized_quantity < min_notional:
            raise ValueError(
                f"notional {quantized_price * quantized_quantity} below minNotional "
                f"{min_notional} for {symbol.upper()}"
            )
        return format(quantized_quantity.normalize(), "f"), format(quantized_price.normalize(), "f")

    def account_balances(self, credentials: BinanceCredentials) -> list[dict[str, Any]]:
        """Fetch non-zero asset balances from the signed /v3/account endpoint."""
        raw = self._signed_request(credentials, "GET", "/v3/account", {})
        balances: list[dict[str, Any]] = []
        for entry in raw.get("balances", []):
            free = float(entry.get("free", 0))
            locked = float(entry.get("locked", 0))
            if free > 0 or locked > 0:
                balances.append({"asset": entry["asset"], "free": free, "locked": locked})
        balances.sort(key=lambda item: item["asset"])
        return balances

    def _limit_order(
        self,
        credentials: BinanceCredentials,
        symbol: str,
        side: str,
        quantity: str | float,
        price: str | float,
        *,
        endpoint: str,
        submitted: bool,
        client_order_id: str | None,
    ) -> dict[str, Any]:
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": normalized_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": _decimal(quantity),
            "price": _decimal(price),
            "newOrderRespType": "RESULT",
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        raw: dict[str, Any] = self._signed_request(credentials, "POST", endpoint, params)
        return {"ok": True, "submitted": submitted, "endpoint": endpoint, "raw": raw}

    def _public_request(self, method: str, endpoint: str, params: dict[str, Any]) -> Any:
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}{endpoint}" + (f"?{query}" if query else "")
        for attempt in range(_MAX_ATTEMPTS):
            request = urllib.request.Request(url, method=method)
            try:
                return self._execute(request, endpoint)
            except BinanceRequestError as exc:
                if attempt == _MAX_ATTEMPTS - 1 or not _retryable(method, exc):
                    raise
                time.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
        raise AssertionError("unreachable")

    def _signed_request(
        self,
        credentials: BinanceCredentials,
        method: str,
        endpoint: str,
        params: dict[str, Any],
    ) -> Any:
        for attempt in range(_MAX_ATTEMPTS):
            signed_params = dict(params)
            # Fresh timestamp and signature per attempt: a retry after a -1021
            # rejection or backoff sleep must not reuse the stale signature.
            signed_params["timestamp"] = int(time.time() * 1000)
            signed_params.setdefault("recvWindow", _RECV_WINDOW_MS)
            query = urllib.parse.urlencode(signed_params)
            signature = hmac.new(
                credentials.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
            )
            request = urllib.request.Request(
                f"{self.base_url}{endpoint}?{query}&signature={signature.hexdigest()}",
                method=method,
                headers={"X-MBX-APIKEY": credentials.api_key},
            )
            try:
                return self._execute(request, endpoint)
            except BinanceRequestError as exc:
                if attempt == _MAX_ATTEMPTS - 1 or not _retryable(method, exc):
                    raise
                time.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
        raise AssertionError("unreachable")

    def _execute(self, request: urllib.request.Request, endpoint: str) -> Any:
        try:
            with self.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8")
            raise BinanceRequestError(
                f"Binance request failed ({endpoint}): HTTP {exc.code}: {body}", status=exc.code
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise BinanceRequestError(f"Binance request failed ({endpoint}): {exc}") from exc
        return json.loads(body) if body else {}


def _quantize_down(value: Decimal, step: Decimal | None) -> Decimal:
    """Floor value to the nearest multiple of step (no-op if step missing/zero)."""
    if step is None or step <= 0:
        return value
    return (value // step) * step


def _decimal(value: str | float) -> str:
    if isinstance(value, str):
        candidate = value.strip()
    else:
        candidate = str(value)
    try:
        decimal = Decimal(candidate)
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc
    if decimal <= 0:
        raise ValueError("decimal values must be > 0")
    return format(decimal.normalize(), "f")
