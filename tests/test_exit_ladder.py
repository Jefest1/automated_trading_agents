from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.config import AppConfig, ExitConfig, Settings, load_config, save_config
from trading_agent.core.exit_ladder import (
    apply_tier_fill,
    build_exit_plan,
    next_ladder_action,
    remaining_quantity,
    update_trail,
)
from trading_agent.core.models import OrderRecord, OrderStatus, Side
from trading_agent.core.storage import Store


def _default_exits() -> ExitConfig:
    # Pin an explicit intraday-width bracket so these ladder-LOGIC tests stay
    # independent of the production swing-config defaults (4% stop / 3%-6% TP /
    # ATR trail). Here we verify the math (tiers, ratchets, % trail), not config.
    return ExitConfig(
        initial_stop_loss_pct=0.01,
        take_profit_tiers=[
            {"profit_pct": 0.015, "size_pct": 0.40},
            {"profit_pct": 0.030, "size_pct": 0.30},
        ],
        trail_pct=0.02,
        trail_atr_mult=None,
    )


class BuildExitPlanTest(unittest.TestCase):
    def test_tiered_plan_has_legs_runner_and_stop(self) -> None:
        plan = build_exit_plan(
            100.0, _default_exits(), fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )
        self.assertTrue(plan.tiered)
        self.assertEqual([leg.tier for leg in plan.legs], [1, 2])
        self.assertAlmostEqual(plan.legs[0].target_price, 101.5)
        self.assertAlmostEqual(plan.legs[1].target_price, 103.0)
        self.assertAlmostEqual(plan.runner_size_pct, 0.30)
        self.assertAlmostEqual(plan.initial_stop_price, 99.0)
        self.assertAlmostEqual(plan.current_stop_price, 99.0)

    def test_disabled_plan_is_single_leg_legacy(self) -> None:
        plan = build_exit_plan(
            100.0,
            ExitConfig(enabled=False),
            fallback_take_profit_pct=0.015,
            fallback_stop_loss_pct=0.01,
        )
        self.assertFalse(plan.tiered)
        self.assertEqual(len(plan.legs), 1)
        self.assertAlmostEqual(plan.legs[0].size_pct, 1.0)
        self.assertAlmostEqual(plan.legs[0].target_price, 101.5)
        self.assertEqual(plan.runner_size_pct, 0.0)


class LadderActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _default_exits()
        self.plan = build_exit_plan(
            100.0, self.cfg, fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )

    def test_quiet_market_yields_no_action(self) -> None:
        action = next_ladder_action(self.plan, 100.8, 1.0)
        self.assertEqual(action.kind, "NONE")

    def test_take_profit_tier_one(self) -> None:
        action = next_ladder_action(self.plan, 101.6, 1.0)
        self.assertEqual(action.kind, "TAKE_TIER")
        self.assertEqual(action.tier, 1)
        self.assertAlmostEqual(action.quantity, 0.40)
        self.assertEqual(action.reason, "TAKE_PROFIT_1")

    def test_stop_out(self) -> None:
        action = next_ladder_action(self.plan, 98.0, 1.0)
        self.assertEqual(action.kind, "STOP_OUT")
        self.assertAlmostEqual(action.quantity, 1.0)
        self.assertEqual(action.reason, "STOP_LOSS")

    def test_ratchet_breakeven_then_lock(self) -> None:
        apply_tier_fill(self.plan, 1, 100.0, self.cfg, filled_qty=0.4)
        self.assertAlmostEqual(self.plan.current_stop_price, 100.0)  # breakeven
        apply_tier_fill(self.plan, 2, 100.0, self.cfg, filled_qty=0.3)
        self.assertAlmostEqual(self.plan.current_stop_price, 101.5)  # locked to TP1
        self.assertTrue(self.plan.runner_active)

    def test_trailing_stop_only_moves_up(self) -> None:
        apply_tier_fill(self.plan, 1, 100.0, self.cfg, filled_qty=0.4)
        apply_tier_fill(self.plan, 2, 100.0, self.cfg, filled_qty=0.3)
        update_trail(self.plan, 110.0, self.cfg)  # high-water 110 -> stop 107.8
        self.assertAlmostEqual(self.plan.current_stop_price, 107.8)
        update_trail(self.plan, 105.0, self.cfg)  # lower mark must not lower the stop
        self.assertAlmostEqual(self.plan.current_stop_price, 107.8)
        self.assertAlmostEqual(remaining_quantity(self.plan, 1.0), 0.30)


class ExitPlanPersistenceTest(unittest.TestCase):
    def test_exit_plan_round_trips(self) -> None:
        with TemporaryDirectory() as tmp:
            config = AppConfig(home=tmp, database_path=str(Path(tmp) / "a.sqlite3"))
            with Store(config.database_path) as store:
                plan = build_exit_plan(
                    100.0, config.exits, fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
                )
                order = OrderRecord(
                    proposal_id="tp_x",
                    mode="testnet",
                    symbol="BTCUSDT",
                    side=Side.BUY,
                    order_type="SPOT_TESTNET_LIMIT_ENTRY",
                    price=100.0,
                    quantity=1.0,
                    take_profit_price=101.5,
                    stop_loss_price=99.0,
                    status=OrderStatus.POSITION_OPEN,
                    exit_plan=plan,
                )
                store.save_order(order)
                reloaded = store.open_positions()[0]
        self.assertIsNotNone(reloaded.exit_plan)
        self.assertTrue(reloaded.exit_plan.tiered)
        self.assertEqual(len(reloaded.exit_plan.legs), 3)

    def test_legacy_row_without_column_loads_single_leg_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "legacy.sqlite3")
            # Simulate a pre-tiered DB: create an orders row, then drop the column
            # is impossible in sqlite; instead insert with NULL exit_plan_json.
            with Store(db) as store:
                store.conn.execute(
                    "INSERT INTO orders (id, proposal_id, mode, symbol, side, order_type, "
                    "price, quantity, take_profit_price, stop_loss_price, status, opened_at, "
                    "entry_fee, exit_fee, realized_pnl, exit_plan_json) VALUES "
                    "('ord_legacy','tp_x','paper','BTCUSDT','BUY','SPOT_LIMIT_ENTRY',"
                    "100.0,1.0,101.5,99.0,'POSITION_OPEN','2026-01-01T00:00:00+00:00',"
                    "0,0,0,NULL)"
                )
                store.conn.commit()
                order = [o for o in store.open_positions() if o.id == "ord_legacy"][0]
        self.assertIsNotNone(order.exit_plan)
        self.assertFalse(order.exit_plan.tiered)
        self.assertAlmostEqual(order.exit_plan.legs[0].target_price, 101.5)
        self.assertAlmostEqual(order.exit_plan.current_stop_price, 99.0)

    def test_migration_adds_exit_plan_column(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "m.sqlite3")
            with Store(db):
                pass
            conn = sqlite3.connect(db)
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(orders)")}
            finally:
                conn.close()
        self.assertIn("exit_plan_json", cols)


class MonitorOpenPositionsTest(unittest.TestCase):
    """The fast bracket monitor is exchange-only after paper execution removal."""

    def test_monitor_is_noop_when_flat(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(str(root / "agent.sqlite3")) as store:
                config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
                from trading_agent.graph import SupervisorRuntime

                runtime = SupervisorRuntime(config, store, settings=Settings())
                summary = runtime.monitor_open_positions()
        self.assertEqual(summary, {"checked": 0, "exits": 0})


class ExitConfigRoundTripTest(unittest.TestCase):
    def test_config_persists_and_merges(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "a.sqlite3"))
            config.exits.trail_pct = 0.03
            config.exits.take_profit_tiers = [{"profit_pct": 0.02, "size_pct": 0.5}]
            save_config(config)
            reloaded = load_config(root)
        self.assertAlmostEqual(reloaded.exits.trail_pct, 0.03)
        self.assertEqual(len(reloaded.exits.take_profit_tiers), 1)
        self.assertAlmostEqual(reloaded.exits.runner_size_pct, 0.5)


if __name__ == "__main__":
    unittest.main()
