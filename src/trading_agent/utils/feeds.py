from __future__ import annotations

import hashlib
import math
from datetime import timedelta

from trading_agent.core.models import MarketSnapshot, utc_now


BASE_PRICES = {
    "BTCUSDT": 100_000.0,
    "ETHUSDT": 4_000.0,
    "SOLUSDT": 180.0,
    "BNBUSDT": 700.0,
}


class SimulatedMarketFeed:
    """Deterministic market feed for repeatable offline simulation/backtests."""

    def snapshot(self, symbol: str, cycle: int) -> MarketSnapshot:
        base = BASE_PRICES.get(symbol, 100.0)
        phase = self._unit(symbol, "phase") * math.tau
        drift = (self._unit(symbol, "drift") - 0.48) * 0.0015
        wave = math.sin(cycle / 4.0 + phase) * 0.008
        pulse = (self._unit(symbol, str(cycle)) - 0.5) * 0.004
        last_price = base * (1 + drift * cycle + wave + pulse)
        spread_bps = 4.0 + self._unit(symbol, "spread", str(cycle)) * 6.0
        bid = last_price * (1 - spread_bps / 20_000)
        ask = last_price * (1 + spread_bps / 20_000)
        volume = base * (1_000 + self._unit(symbol, "volume", str(cycle)) * 5_000)
        observed_at = (utc_now() + timedelta(minutes=15 * cycle)).isoformat()
        return MarketSnapshot(
            symbol=symbol,
            observed_at=observed_at,
            last_price=round(last_price, 8),
            bid_price=round(bid, 8),
            ask_price=round(ask, 8),
            volume_24h=round(volume, 2),
            source="simulated",
        )

    @staticmethod
    def _unit(*parts: str) -> float:
        digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
