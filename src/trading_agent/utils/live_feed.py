from __future__ import annotations

import json
from typing import Any

from trading_agent.core.logging import get_logger
from trading_agent.core.models import MarketSnapshot, utc_iso
from trading_agent.exchange import BinanceSpotAdapter
from trading_agent.utils.binance_skills import MAJOR_TOKEN_CONTRACTS, BinanceSkillRegistry
from trading_agent.utils.feeds import SimulatedMarketFeed

LOGGER = get_logger("live_feed")

_PRICE_KEYS = ("lastPrice", "last_price", "price", "close", "c")
_BID_KEYS = ("bidPrice", "bid_price", "bid", "b")
_ASK_KEYS = ("askPrice", "ask_price", "ask", "a")
_VOLUME_KEYS = ("quoteVolume", "volume_24h", "volume24h", "volume", "v")
_DEFAULT_SPREAD_BPS = 2.0


class BinanceLiveFeed:
    """Live market snapshots, free-first.

    Source order per snapshot:
    1. Binance Skills Hub ``query-token-info dynamic`` (agents' own skill).
    2. Public Binance REST ticker via the existing exchange adapter.
    3. Deterministic simulated feed (offline fallback).

    The source actually used is recorded in ``last_source`` so callers (REPL
    header, evidence payloads) can show LIVE vs SIM provenance.
    """

    def __init__(
        self,
        registry: BinanceSkillRegistry,
        fallback: SimulatedMarketFeed,
        *,
        adapter: BinanceSpotAdapter | None = None,
    ) -> None:
        self.registry = registry
        self.fallback = fallback
        self.adapter = adapter
        self.last_source: dict[str, str] = {}

    def snapshot(self, symbol: str, cycle: int) -> MarketSnapshot:
        snapshot = self._snapshot_from_skill(symbol)
        if snapshot is None:
            snapshot = self._snapshot_from_rest(symbol)
        if snapshot is None:
            self.last_source[symbol] = "simulated"
            return self.fallback.snapshot(symbol, cycle)
        return snapshot

    def _snapshot_from_skill(self, symbol: str) -> MarketSnapshot | None:
        # query-token-info is keyed by (chainId, contractAddress); symbol-style
        # params are rejected upstream ("illegal parameter").
        token = symbol.upper().removesuffix("USDT")
        contract = MAJOR_TOKEN_CONTRACTS.get(token)
        if contract is None:
            return None
        params = json.dumps(
            {"chainId": contract["chainId"], "contractAddress": contract["contractAddress"]},
            sort_keys=True,
        )
        try:
            result = self.registry.run_read_only_cli("query-token-info", "dynamic", params)
        except Exception as exc:
            LOGGER.debug("skill feed unavailable symbol=%s error=%s", symbol, exc)
            return None
        if result.returncode != 0 or not result.stdout.strip():
            LOGGER.debug(
                "skill feed non-zero symbol=%s returncode=%s stderr=%s",
                symbol,
                result.returncode,
                result.stderr[:200],
            )
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            LOGGER.debug("skill feed returned non-JSON symbol=%s", symbol)
            return None
        snapshot = self._snapshot_from_payload(symbol, payload)
        if snapshot is not None:
            self.last_source[symbol] = "binance-skills-hub"
        return snapshot

    def _snapshot_from_rest(self, symbol: str) -> MarketSnapshot | None:
        if self.adapter is None:
            return None
        try:
            ticker = self.adapter.ticker_price(symbol)
            price = float(ticker["price"])
        except Exception as exc:
            LOGGER.debug("rest feed unavailable symbol=%s error=%s", symbol, exc)
            return None
        self.last_source[symbol] = "binance-rest"
        half_spread = price * _DEFAULT_SPREAD_BPS / 20_000
        return MarketSnapshot(
            symbol=symbol.upper(),
            observed_at=utc_iso(),
            last_price=round(price, 8),
            bid_price=round(price - half_spread, 8),
            ask_price=round(price + half_spread, 8),
            volume_24h=0.0,
        )

    def _snapshot_from_payload(self, symbol: str, payload: Any) -> MarketSnapshot | None:
        price = _find_number(payload, _PRICE_KEYS)
        if price is None or price <= 0:
            return None
        bid = _find_number(payload, _BID_KEYS)
        ask = _find_number(payload, _ASK_KEYS)
        volume = _find_number(payload, _VOLUME_KEYS) or 0.0
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            half_spread = price * _DEFAULT_SPREAD_BPS / 20_000
            bid = price - half_spread
            ask = price + half_spread
        return MarketSnapshot(
            symbol=symbol.upper(),
            observed_at=utc_iso(),
            last_price=round(price, 8),
            bid_price=round(bid, 8),
            ask_price=round(ask, 8),
            volume_24h=round(volume, 2),
        )


def _find_number(payload: Any, keys: tuple[str, ...], depth: int = 0) -> float | None:
    """Defensively locate the first numeric value under any of ``keys``."""
    if depth > 4:
        return None
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                value = _coerce_float(payload[key])
                if value is not None:
                    return value
        for value in payload.values():
            found = _find_number(value, keys, depth + 1)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload[:5]:
            found = _find_number(item, keys, depth + 1)
            if found is not None:
                return found
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
