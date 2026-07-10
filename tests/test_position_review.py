from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from trading_agent.agents.position_review import PositionReviewAgent
from trading_agent.core.config import AppConfig
from trading_agent.core.models import (
    REVIEW_CANCEL_CANDIDATE,

    REVIEW_HOLD,
    REVIEW_KEEP,
    LevelMap,
    MarketSnapshot,
    OrderRecord,
    OrderStatus,
    Side,
)


def _snap(symbol: str, price: float) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        observed_at=datetime.now(UTC).isoformat(),
        last_price=price,
        bid_price=price - 0.01,
        ask_price=price + 0.01,
        volume_24h=1_000.0,
    )


def _level_map(symbol: str, regime: str, price: float) -> LevelMap:
    return LevelMap(symbol=symbol, current_price=price, regime=regime, support_zones=[], resistance_zones=[])


def _position(symbol: str = "BTCUSDT", *, entry: float = 100.0, age_hours: float = 10.0,
              status: OrderStatus = OrderStatus.POSITION_OPEN) -> OrderRecord:
    opened = (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat()
    order = OrderRecord(
        proposal_id="tp_x",
        mode="testnet",
        symbol=symbol,
        side=Side.BUY,
        order_type="SPOT_LIMIT_ENTRY",
        price=entry,
        quantity=1.0,
        take_profit_price=entry * 1.06,
        stop_loss_price=entry * 0.96,
        status=status,
        opened_at=opened,
    )
    order.executed_qty = 1.0 if status == OrderStatus.POSITION_OPEN else 0.0
    order.avg_fill_price = entry if status == OrderStatus.POSITION_OPEN else None
    return order


class PositionReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = PositionReviewAgent()
        self.config = AppConfig()  # min_hold_hours = 4.0 default

    def _review_one(self, order, price, regime):
        snaps = {order.symbol: _snap(order.symbol, price)}
        lmaps = {order.symbol: _level_map(order.symbol, regime, price)}
        return self.agent.review([order], snaps, lmaps, self.config)[0]

    def test_healthy_open_long_holds(self) -> None:
        order = _position(entry=100.0, age_hours=10.0)
        r = self._review_one(order, price=104.0, regime="uptrend")
        self.assertEqual(r.recommended_action, REVIEW_HOLD)
        self.assertTrue(r.min_hold_satisfied)

    def test_downtrend_underwater_position_still_holds_no_touch(self) -> None:
        # No-touch policy: even a downtrending, underwater position is HELD (the
        # stop protects; only a critical news catalyst may close it); but the
        # reason flags it for the news sentry's attention.
        order = _position(entry=100.0, age_hours=10.0)
        r = self._review_one(order, price=95.0, regime="downtrend")
        self.assertEqual(r.recommended_action, REVIEW_HOLD)
        self.assertIn("news sentry", r.reason)
        self.assertLess(r.unrealized_pnl_pct, 0)

    def test_downtrend_underwater_within_min_hold_holds(self) -> None:
        order = _position(entry=100.0, age_hours=1.0)  # < 4h cooldown
        r = self._review_one(order, price=95.0, regime="downtrend")
        self.assertEqual(r.recommended_action, REVIEW_HOLD)
        self.assertFalse(r.min_hold_satisfied)

    def test_resting_bid_kept_when_not_bearish(self) -> None:
        order = _position(status=OrderStatus.ENTRY_OPEN, age_hours=1.0)
        r = self._review_one(order, price=99.0, regime="range")
        self.assertEqual(r.recommended_action, REVIEW_KEEP)

    def test_resting_bid_flagged_cancel_on_downtrend(self) -> None:
        order = _position(status=OrderStatus.ENTRY_OPEN, age_hours=1.0)
        r = self._review_one(order, price=99.0, regime="downtrend")
        self.assertEqual(r.recommended_action, REVIEW_CANCEL_CANDIDATE)

    def test_min_hold_boundary(self) -> None:
        fresh = self._review_one(_position(age_hours=3.9), 104.0, "uptrend")
        aged = self._review_one(_position(age_hours=4.1), 104.0, "uptrend")
        self.assertFalse(fresh.min_hold_satisfied)
        self.assertTrue(aged.min_hold_satisfied)


if __name__ == "__main__":
    unittest.main()
