from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_agent.agents.strategy import StrategyAgent
from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.decision import DecisionAction, SupervisorDecision
from trading_agent.core.models import (
    EvidenceRecord,
    ExitLeg,
    ExitPlan,
    MarketSnapshot,
    OrderRecord,
    OrderStatus,
    Side,
    utc_iso,
)
from trading_agent.core.pnl import unrealized_pnl
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime
from trading_agent.graph.nodes import CycleNodes
from trading_agent.utils import market_data
from trading_agent.utils.mcp_tools import default_mcp_servers


def _open_order(price: float = 100.0, qty: float = 2.0) -> OrderRecord:
    order = OrderRecord(
        proposal_id="tp_x",
        mode="testnet",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type="SPOT_LIMIT_ENTRY",
        price=price,
        quantity=qty,
        take_profit_price=price * 1.015,
        stop_loss_price=price * 0.99,
        status=OrderStatus.POSITION_OPEN,
    )
    order.executed_qty = qty
    return order


class UnrealizedPnlTest(unittest.TestCase):
    def test_open_position_marks_to_price(self) -> None:
        up, up_pct = unrealized_pnl(_open_order(100.0, 2.0), 110.0)
        self.assertEqual(up, 20.0)
        self.assertEqual(up_pct, 10.0)

    def test_no_price_or_closed_is_none(self) -> None:
        self.assertEqual(unrealized_pnl(_open_order(), None), (None, None))
        closed = _open_order()
        closed.status = OrderStatus.CLOSED
        self.assertEqual(unrealized_pnl(closed, 110.0), (None, None))

    def test_scaled_out_position_marks_only_held_portion(self) -> None:
        # After TP1 banks 40%, unrealized is on the remaining 60%, not the full size.
        order = _open_order(100.0, 1.0)
        order.exit_plan = ExitPlan(
            legs=[
                ExitLeg(tier=1, target_price=103.0, size_pct=0.4, filled=True, filled_qty=0.4),
                ExitLeg(tier=2, target_price=106.0, size_pct=0.3),
            ],
            initial_stop_price=96.0,
            current_stop_price=100.0,
            high_water_price=110.0,
            runner_size_pct=0.3,
            tiered=True,
        )
        up, up_pct = unrealized_pnl(order, 110.0)
        self.assertEqual(up, 6.0)  # (110-100) * (1.0 - 0.4 held), NOT * 1.0 = 10.0
        self.assertEqual(up_pct, 10.0)  # per-unit pct is unchanged


class CachedPricesTest(unittest.TestCase):
    def test_cache_hits_within_ttl_then_refetches(self) -> None:
        with TemporaryDirectory() as tmp:
            with Store(Path(tmp) / "agent.sqlite3") as store:
                with patch.object(market_data, "current_prices", return_value={"BTCUSDT": 5.0}) as cp:
                    a = market_data.cached_current_prices(["BTCUSDT"], store, ttl=300)
                    b = market_data.cached_current_prices(["BTCUSDT"], store, ttl=300)
                    self.assertEqual(a, {"BTCUSDT": 5.0})
                    self.assertEqual(b, {"BTCUSDT": 5.0})
                    self.assertEqual(cp.call_count, 1)  # second call served from cache
                with patch.object(market_data, "current_prices", return_value={"BTCUSDT": 6.0}) as cp2:
                    c = market_data.cached_current_prices(["BTCUSDT"], store, ttl=0)
                    self.assertEqual(c, {"BTCUSDT": 6.0})
                    self.assertEqual(cp2.call_count, 1)  # ttl=0 forces refetch


class MakerPullbackEntryTest(unittest.TestCase):
    def test_strategy_limit_rests_below_bid_by_atr(self) -> None:
        config = AppConfig()  # entry_atr_mult 0.3, min 5 bps, max 1.5%
        snapshot = MarketSnapshot(
            symbol="BTCUSDT",
            observed_at=utc_iso(),
            last_price=100.0,
            bid_price=100.0,
            ask_price=100.1,
            volume_24h=1000.0,
            atr=2.0,
        )
        evidence = [
            EvidenceRecord(
                agent="market_data_agent",
                source="test",
                symbol="BTCUSDT",
                kind="price_order_book",
                observed_at=utc_iso(),
                score=0.5,
                confidence=0.8,
                payload={},
            )
        ]
        proposals = StrategyAgent().propose({"BTCUSDT": snapshot}, evidence, config)
        self.assertEqual(len(proposals), 1)
        self.assertLess(proposals[0].price, snapshot.bid_price)
        self.assertAlmostEqual(proposals[0].price, 100.0 - 0.3 * 2.0, places=6)


class McpDefaultsTest(unittest.TestCase):
    def test_coingecko_dropped_helium_and_fxmacro_present(self) -> None:
        names = {s.name for s in default_mcp_servers()}
        self.assertNotIn("coingecko", names)
        self.assertIn("helium_news", names)
        self.assertIn("fxmacrodata", names)


class ContextAndCloseTest(unittest.TestCase):
    def _runtime(self, root: Path, store: Store) -> SupervisorRuntime:
        config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
        return SupervisorRuntime(config, store, settings=Settings())

    def test_cycle_context_carries_current_date(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                ctx = self._runtime(root, store).cycle_context()
        self.assertIn("now", ctx)
        self.assertTrue(ctx["now"]["date"])
        self.assertIn("weekday", ctx["now"])
        self.assertIn("daily_brief", ctx)

    def test_close_of_stale_target_is_noop_not_error(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                nodes = CycleNodes(self._runtime(root, store))
                decision = SupervisorDecision(
                    action=DecisionAction.CLOSE,
                    symbol="BTCUSDT",
                    target_order_id="ord_does_not_exist",
                    rationale="stale",
                )
                # No open order with that id: must not raise.
                nodes._execute_close(decision, {}, "run_test")


if __name__ == "__main__":
    unittest.main()
