from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.models import OrderRecord, OrderStatus, Side, utc_iso
from trading_agent.core.reflection import build_reflection, realized_r
from trading_agent.core.storage import Store


def closed_order(
    *, symbol: str = "SOLUSDT", entry: float = 100.0, exit_price: float = 102.0,
    stop: float = 99.0, qty: float = 1.0, pnl: float = 2.0, reason: str = "TAKE_PROFIT",
) -> OrderRecord:
    return OrderRecord(
        proposal_id="tp_x",
        mode="testnet",
        symbol=symbol,
        side=Side.BUY,
        order_type="SPOT_LIMIT_ENTRY",
        price=entry,
        quantity=qty,
        take_profit_price=exit_price,
        stop_loss_price=stop,
        status=OrderStatus.CLOSED,
        opened_at="2026-06-18T10:00:00+00:00",
        closed_at="2026-06-18T12:00:00+00:00",
        exit_price=exit_price,
        exit_reason=reason,
        realized_pnl=pnl,
        executed_qty=qty,
        avg_fill_price=entry,
    )


class ReflectionTest(unittest.TestCase):
    def test_realized_r_uses_initial_risk(self) -> None:
        # entry 100, stop 99 -> risk 1/unit; pnl +2 on 1 unit -> +2R.
        order = closed_order(entry=100.0, stop=99.0, qty=1.0, pnl=2.0)
        self.assertAlmostEqual(realized_r(order), 2.0)

    def test_realized_r_none_without_valid_risk(self) -> None:
        order = closed_order(entry=100.0, stop=100.0)  # zero risk distance
        self.assertIsNone(realized_r(order))

    def test_build_reflection_classifies_outcome(self) -> None:
        win = build_reflection(closed_order(pnl=2.0))
        loss = build_reflection(closed_order(pnl=-1.0, exit_price=99.0, reason="STOP_LOSS"))
        self.assertEqual(win.outcome, "win")
        self.assertEqual(loss.outcome, "loss")
        self.assertEqual(win.holding_minutes, 120.0)
        self.assertIn("SOLUSDT", win.lesson)

    def test_save_is_idempotent_per_order(self) -> None:
        with TemporaryDirectory() as tmp:
            with Store(Path(tmp) / "agent.sqlite3") as store:
                order = closed_order()
                store.save_reflection(build_reflection(order))
                store.save_reflection(build_reflection(order))  # same order_id
                self.assertEqual(len(store.recent_reflections(10)), 1)

    def test_trade_stats_aggregate(self) -> None:
        with TemporaryDirectory() as tmp:
            with Store(Path(tmp) / "agent.sqlite3") as store:
                win = closed_order(symbol="SOLUSDT", pnl=2.0)
                loss = closed_order(symbol="BTCUSDT", entry=100.0, stop=99.0, exit_price=99.0, pnl=-1.0, reason="STOP_LOSS")
                for order in (win, loss):
                    store.save_order(order)
                    store.save_reflection(build_reflection(order))
                stats = store.trade_stats()
        self.assertEqual(stats["closed_trades"], 2)
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 1)
        self.assertAlmostEqual(stats["win_rate"], 0.5)
        # +2R and -1R -> average +0.5R.
        self.assertAlmostEqual(stats["avg_realized_r"], 0.5)


if __name__ == "__main__":
    unittest.main()
