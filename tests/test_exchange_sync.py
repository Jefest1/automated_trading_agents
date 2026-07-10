from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from trading_agent.core.config import ExitConfig, Settings
from trading_agent.core.exchange_sync import ExchangeReconciler
from trading_agent.core.exit_ladder import build_exit_plan
from trading_agent.core.models import MarketSnapshot, OrderRecord, OrderStatus, Side, utc_iso
from trading_agent.core.storage import Store
from trading_agent.exchange import BinanceCredentials


def _intraday_exits() -> ExitConfig:
    # Explicit intraday-width bracket so the resting-leg / stop-out LOGIC tests
    # stay independent of the production swing-config defaults (4% stop / 3%-6% TP).
    return ExitConfig(
        initial_stop_loss_pct=0.01,
        take_profit_tiers=[
            {"profit_pct": 0.015, "size_pct": 0.40},
            {"profit_pct": 0.030, "size_pct": 0.30},
        ],
        trail_pct=0.02,
        trail_atr_mult=None,
    )


class FakeAdapter:
    """Scriptable stand-in for BinanceSpotAdapter."""

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.trades: dict[str, list[dict[str, Any]]] = {}
        self.submitted: list[dict[str, Any]] = []
        self.canceled: list[str] = []
        self.bid_price = 100.0
        self._next_order_id = 9000

    def get_order(self, credentials, symbol, *, order_id=None, client_order_id=None):
        key = str(order_id) if order_id is not None else str(client_order_id)
        if key not in self.orders:
            raise RuntimeError('Binance request failed (/v3/order): HTTP 400: {"code":-2013,"msg":"Order does not exist."}')
        return self.orders[key]

    def get_my_trades(self, credentials, symbol, *, order_id=None, start_time=None, limit=1000):
        return self.trades.get(str(order_id), [])

    def submit_limit_order(self, credentials, symbol, side, quantity, price, *, client_order_id=None):
        self._next_order_id += 1
        record = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "client_order_id": client_order_id,
            "orderId": self._next_order_id,
        }
        self.submitted.append(record)
        return {"ok": True, "submitted": True, "raw": {"orderId": self._next_order_id, "status": "NEW"}}

    def cancel_order(self, credentials, symbol, *, order_id=None, client_order_id=None):
        self.canceled.append(str(order_id or client_order_id))
        return {"status": "CANCELED"}

    def quantize_order(self, symbol, quantity, price):
        return str(quantity), str(price)

    def book_ticker(self, symbol):
        return {"symbol": symbol, "bid_price": str(self.bid_price), "ask_price": str(self.bid_price + 1)}


def snapshot(symbol: str, last_price: float) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        observed_at=utc_iso(),
        last_price=last_price,
        bid_price=last_price - 0.5,
        ask_price=last_price + 0.5,
        volume_24h=1000.0,
    )


def testnet_order(**overrides: Any) -> OrderRecord:
    defaults: dict[str, Any] = dict(
        proposal_id="tp_1",
        mode="testnet",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type="SPOT_TESTNET_LIMIT_ENTRY",
        price=100.0,
        quantity=1.0,
        take_profit_price=105.0,
        stop_loss_price=95.0,
        exchange_order_id="42",
        client_order_id="ta42",
    )
    defaults.update(overrides)
    return OrderRecord(**defaults)


class ExchangeReconcilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "test.sqlite3")
        self.adapter = FakeAdapter()
        self.reconciler = ExchangeReconciler(
            self.store,
            Settings(),
            adapter=self.adapter,  # type: ignore[arg-type]
            credentials=BinanceCredentials(api_key="k", api_secret="s"),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_entry_fill_promotes_to_position_open_and_ingests_fills(self) -> None:
        order = testnet_order()
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42,
            "status": "FILLED",
            "executedQty": "1.0",
            "cummulativeQuoteQty": "100.0",
        }
        self.adapter.trades["42"] = [
            {"id": 7, "orderId": 42, "price": "100.0", "qty": "1.0", "quoteQty": "100.0",
             "commission": "0.1", "commissionAsset": "USDT", "time": 1718000000000}
        ]

        updated = self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 101.0)})

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].status, OrderStatus.POSITION_OPEN)
        self.assertEqual(updated[0].exchange_status, "FILLED")
        self.assertEqual(updated[0].executed_qty, 1.0)
        self.assertEqual(updated[0].avg_fill_price, 100.0)
        fills = self.store.fills_for_order(order.id)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].commission, 0.1)

    def test_fill_ingestion_is_idempotent(self) -> None:
        order = testnet_order()
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }
        self.adapter.trades["42"] = [
            {"id": 7, "orderId": 42, "price": "100.0", "qty": "1.0", "quoteQty": "100.0",
             "commission": "0.1", "commissionAsset": "USDT", "time": 1718000000000}
        ]
        self.reconciler.reconcile({})
        order_after = [o for o in self.store.open_positions() if o.id == order.id][0]
        self.reconciler.sync_order(order_after)
        self.assertEqual(len(self.store.fills_for_order(order.id)), 1)

    def test_take_profit_rests_as_real_order_on_open(self) -> None:
        # Resting-order model: the take-profit is placed on the book as soon as the
        # position is open (so the venue fills it intra-second), not only when our
        # poll happens to see the touch.
        order = testnet_order(status=OrderStatus.POSITION_OPEN, executed_qty=1.0)
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }

        # Price below the TP target: the resting sell is still placed at the target.
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 101.0)})

        self.assertEqual(len(self.adapter.submitted), 1)
        self.assertEqual(self.adapter.submitted[0]["side"], "SELL")
        self.assertEqual(float(self.adapter.submitted[0]["price"]), 105.0)
        synced = self.store.open_positions()[0]
        # The resting order is tracked on the take-profit leg, not the order-level
        # exit pointer (which is reserved for the stop-out / operator close).
        self.assertIsNotNone(synced.exit_plan)
        self.assertIsNotNone(synced.exit_plan.legs[0].exit_order_exchange_id)

    def test_exit_fill_closes_order_with_fill_based_pnl(self) -> None:
        order = testnet_order(
            status=OrderStatus.POSITION_OPEN,
            executed_qty=1.0,
            exit_order_exchange_id="43",
            exit_client_order_id="tx43",
            exit_reason="TAKE_PROFIT",
        )
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }
        self.adapter.orders["43"] = {
            "orderId": 43, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "105.0",
        }
        self.adapter.trades["42"] = [
            {"id": 7, "orderId": 42, "price": "100.0", "qty": "1.0", "quoteQty": "100.0",
             "commission": "0.1", "commissionAsset": "USDT", "time": 1718000000000}
        ]
        self.adapter.trades["43"] = [
            {"id": 8, "orderId": 43, "price": "105.0", "qty": "1.0", "quoteQty": "105.0",
             "commission": "0.105", "commissionAsset": "USDT", "time": 1718000100000}
        ]

        self.reconciler.reconcile({})

        closed = self.store.per_trade_pnl()
        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0]["realized_pnl"], 105.0 - 100.0 - 0.205)
        self.assertEqual(closed[0]["exit_reason"], "TAKE_PROFIT")
        self.assertFalse(bool(closed[0]["pnl_estimated"]))

    def test_vanished_order_marked_unknown_without_crash(self) -> None:
        order = testnet_order(exchange_order_id="404", client_order_id=None)
        self.store.save_order(order)

        updated = self.reconciler.reconcile({})

        self.assertEqual(updated[0].exchange_status, "UNKNOWN")
        self.assertEqual(updated[0].status, OrderStatus.ENTRY_OPEN)

    def test_close_position_cancels_unfilled_entry(self) -> None:
        order = testnet_order()
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "NEW", "executedQty": "0", "cummulativeQuoteQty": "0",
        }

        closed = self.reconciler.close_position(order, reason="OPERATOR_CLOSE")

        self.assertEqual(closed.status, OrderStatus.CANCELED)
        self.assertIn("42", self.adapter.canceled)
        self.assertEqual(self.adapter.submitted, [])

    def test_close_position_sells_open_position_at_bid(self) -> None:
        order = testnet_order(status=OrderStatus.POSITION_OPEN, executed_qty=1.0)
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }
        self.adapter.bid_price = 102.0

        closed = self.reconciler.close_position(order)

        self.assertEqual(len(self.adapter.submitted), 1)
        self.assertEqual(float(self.adapter.submitted[0]["price"]), 102.0)
        self.assertEqual(closed.closed_by, "OPERATOR_CLOSE")
        self.assertIsNotNone(closed.exit_order_exchange_id)

    def test_paper_orders_are_ignored(self) -> None:
        order = testnet_order(mode="paper")
        self.store.save_order(order)
        self.assertEqual(self.reconciler.reconcile({}), [])

    def test_pending_submit_adopts_live_order(self) -> None:
        # Crash between submit and DB save: the venue knows the client id, the
        # local row is still PENDING_SUBMIT; the reconciler must adopt it.
        order = testnet_order(
            status=OrderStatus.PENDING_SUBMIT,
            exchange_order_id=None,
            client_order_id="taorphan1",
        )
        self.store.save_order(order)
        live = {"orderId": 77, "status": "NEW", "executedQty": "0", "cummulativeQuoteQty": "0"}
        self.adapter.orders["taorphan1"] = live
        self.adapter.orders["77"] = live

        updated = self.reconciler.reconcile({})

        self.assertEqual(len(updated), 1)
        reloaded = [o for o in self.store.open_positions() if o.id == order.id][0]
        self.assertEqual(reloaded.status, OrderStatus.ENTRY_OPEN)
        self.assertEqual(reloaded.exchange_order_id, "77")

    def test_pending_submit_unknown_to_venue_discarded_after_grace(self) -> None:
        stale_opened = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        order = testnet_order(
            status=OrderStatus.PENDING_SUBMIT,
            exchange_order_id=None,
            client_order_id="tanever1",
            opened_at=stale_opened,
        )
        self.store.save_order(order)

        self.reconciler.reconcile({})

        self.assertEqual(self.store.open_positions(), [])
        row = [o for o in self.store.all_orders() if o["id"] == order.id][0]
        self.assertEqual(row["status"], "CANCELED")
        self.assertEqual(row["closed_by"], "SUBMIT_FAILED")

    def test_pending_submit_within_grace_is_left_alone(self) -> None:
        # The submit may still be in flight; do not declare it failed yet.
        order = testnet_order(
            status=OrderStatus.PENDING_SUBMIT,
            exchange_order_id=None,
            client_order_id="tainflight1",
        )
        self.store.save_order(order)

        updated = self.reconciler.reconcile({})

        self.assertEqual(updated, [])
        reloaded = [o for o in self.store.open_positions() if o.id == order.id][0]
        self.assertEqual(reloaded.status, OrderStatus.PENDING_SUBMIT)

    def test_terminal_partial_exit_resubmits_remaining_quantity(self) -> None:
        # An expired exit with partial fills must not wedge the position; the
        # bracket monitor resubmits a sell for the unsold remainder.
        order = testnet_order(
            status=OrderStatus.POSITION_OPEN,
            executed_qty=1.0,
            exit_order_exchange_id="43",
            exit_client_order_id="tx43",
            exit_reason="TAKE_PROFIT",
        )
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }
        self.adapter.orders["43"] = {
            "orderId": 43, "status": "EXPIRED", "executedQty": "0.4", "cummulativeQuoteQty": "42.0",
        }
        self.adapter.trades["42"] = [
            {"id": 7, "orderId": 42, "price": "100.0", "qty": "1.0", "quoteQty": "100.0",
             "commission": "0", "commissionAsset": "USDT", "time": 1718000000000}
        ]
        self.adapter.trades["43"] = [
            {"id": 9, "orderId": 43, "price": "105.0", "qty": "0.4", "quoteQty": "42.0",
             "commission": "0", "commissionAsset": "USDT", "time": 1718000100000}
        ]

        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 106.0)})

        reloaded = [o for o in self.store.open_positions() if o.id == order.id][0]
        self.assertEqual(reloaded.status, OrderStatus.POSITION_OPEN)
        self.assertEqual(len(self.adapter.submitted), 1)
        self.assertEqual(self.adapter.submitted[0]["side"], "SELL")
        self.assertAlmostEqual(float(self.adapter.submitted[0]["quantity"]), 0.6)


class TieredRestingExitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "test.sqlite3")
        self.adapter = FakeAdapter()
        self.reconciler = ExchangeReconciler(
            self.store,
            Settings(),
            adapter=self.adapter,  # type: ignore[arg-type]
            credentials=BinanceCredentials(api_key="k", api_secret="s"),
            exit_config=_intraday_exits(),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _tiered_order(self, **overrides):
        plan = build_exit_plan(
            100.0, _intraday_exits(), fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )
        defaults = dict(
            status=OrderStatus.POSITION_OPEN, executed_qty=1.0, avg_fill_price=100.0, exit_plan=plan
        )
        defaults.update(overrides)
        order = testnet_order(**defaults)
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }
        return order

    def test_both_take_profit_tiers_rest_on_open_runner_left_alone(self) -> None:
        self._tiered_order()
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 100.5)})
        sells = [s for s in self.adapter.submitted if s["side"] == "SELL"]
        self.assertEqual(len(sells), 2)  # TP1 + TP2 rest; runner (30%) is not rested
        prices = sorted(float(s["price"]) for s in sells)
        qtys = sorted(float(s["quantity"]) for s in sells)
        self.assertEqual(prices, [101.5, 103.0])
        self.assertEqual(qtys, [0.3, 0.4])
        plan = self.store.open_positions()[0].exit_plan
        self.assertIsNotNone(plan.legs[0].exit_order_exchange_id)
        self.assertIsNotNone(plan.legs[1].exit_order_exchange_id)

    def test_filled_leg_with_lost_binding_is_adopted_not_resold(self) -> None:
        # Regression: a TP resting below market fills instantly; if the leg
        # binding was lost (crash/exception before save), re-placing must ADOPT
        # the terminal-FILLED order by its deterministic client id; never
        # submit a duplicate sell.
        order = self._tiered_order()
        tier1_client_id = f"tx{order.id.replace('_', '')[:20]}t1"
        self.adapter.orders[tier1_client_id] = {
            "orderId": 7001, "status": "FILLED",
            "executedQty": "0.4", "cummulativeQuoteQty": "40.6",
        }
        self.adapter.orders["7001"] = self.adapter.orders[tier1_client_id]
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 100.5)})
        # No duplicate SELL for tier 1: the only submitted sell is tier 2.
        tier1_sells = [
            s for s in self.adapter.submitted
            if s["side"] == "SELL" and s.get("client_order_id") == tier1_client_id
        ]
        self.assertEqual(tier1_sells, [])
        plan = self.store.open_positions()[0].exit_plan
        leg1 = plan.legs[0]
        self.assertEqual(leg1.exit_order_exchange_id, "7001")  # adopted
        self.assertTrue(leg1.filled)  # fill ingested via sync, stop ratcheted
        self.assertGreaterEqual(plan.current_stop_price, 100.0)  # breakeven after TP1

    def test_partially_filled_canceled_leg_does_not_rearm_full_size(self) -> None:
        # A terminal order WITH partial fills must not re-rest its full tier size
        # (that would re-sell base already sold); it books what it got instead.
        plan = build_exit_plan(
            100.0, _intraday_exits(), fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )
        plan.legs[0].exit_order_exchange_id = "301"
        plan.legs[0].exit_client_order_id = "cx301"
        order = testnet_order(
            status=OrderStatus.POSITION_OPEN, executed_qty=1.0, avg_fill_price=100.0, exit_plan=plan
        )
        self.store.save_order(order)
        self.adapter.orders["42"] = {
            "orderId": 42, "status": "FILLED", "executedQty": "1.0", "cummulativeQuoteQty": "100.0",
        }
        self.adapter.orders["301"] = {
            "orderId": 301, "status": "CANCELED",
            "executedQty": "0.2", "cummulativeQuoteQty": "20.3",
        }
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 100.5)})
        reloaded_plan = self.store.open_positions()[0].exit_plan
        leg1 = reloaded_plan.legs[0]
        self.assertTrue(leg1.filled)  # booked with its partial, not rearmed
        tier1_resells = [
            s for s in self.adapter.submitted
            if s["side"] == "SELL" and float(s["quantity"]) == 0.4
        ]
        self.assertEqual(tier1_resells, [])

    def test_resting_tier_fill_ratchets_stop_to_breakeven(self) -> None:
        plan = build_exit_plan(
            100.0, _intraday_exits(), fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )
        plan.legs[0].exit_order_exchange_id = "201"
        plan.legs[0].exit_client_order_id = "tx201"
        self._tiered_order(exit_plan=plan)
        self.adapter.orders["201"] = {
            "orderId": 201, "status": "FILLED", "executedQty": "0.4", "cummulativeQuoteQty": "40.6",
        }
        self.adapter.trades["201"] = [
            {"id": 11, "orderId": 201, "price": "101.5", "qty": "0.4", "quoteQty": "40.6",
             "commission": "0", "commissionAsset": "USDT", "time": 1718000000000}
        ]
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 101.6)})
        reloaded = self.store.open_positions()[0]
        self.assertTrue(reloaded.exit_plan.legs[0].filled)
        self.assertAlmostEqual(reloaded.exit_plan.current_stop_price, 100.0)  # breakeven

    def test_partial_tp_fill_realizes_positive_not_phantom_loss(self) -> None:
        # Regression: a partial TP1 scale-out (40% of the entry) must realize PnL
        # on the SOLD portion only; NOT book the full entry cost against the
        # partial exit, which produced a large phantom loss (the -19 on live SOL).
        plan = build_exit_plan(
            100.0, _intraday_exits(), fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )
        plan.legs[0].exit_order_exchange_id = "201"
        plan.legs[0].exit_client_order_id = "tx201"
        self._tiered_order(exit_plan=plan)
        # Entry fill (1.0 @ 100) so realized PnL has a cost basis to prorate.
        self.adapter.trades["42"] = [
            {"id": 1, "orderId": 42, "price": "100.0", "qty": "1.0", "quoteQty": "100.0",
             "commission": "0.1", "commissionAsset": "USDT", "time": 1718000000000}
        ]
        # TP1 resting sell (0.4 @ 101.5) fills; runner stays open.
        self.adapter.orders["201"] = {
            "orderId": 201, "status": "FILLED", "executedQty": "0.4", "cummulativeQuoteQty": "40.6",
        }
        self.adapter.trades["201"] = [
            {"id": 11, "orderId": 201, "price": "101.5", "qty": "0.4", "quoteQty": "40.6",
             "commission": "0.04", "commissionAsset": "USDT", "time": 1718000001000}
        ]
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 101.6)})
        reloaded = self.store.open_positions()[0]
        self.assertEqual(reloaded.status, OrderStatus.POSITION_OPEN)  # runner still open
        self.assertGreater(reloaded.realized_pnl, 0.0)
        # 40.6 proceeds − cost basis 0.4×100 (40) − fees (0.04 prorated entry + 0.04 exit) = 0.52
        self.assertAlmostEqual(reloaded.realized_pnl, 40.6 - 40.0 - (0.04 + 0.04), places=6)

    def test_stop_out_cancels_resting_tps_then_market_sells(self) -> None:
        plan = build_exit_plan(
            100.0, _intraday_exits(), fallback_take_profit_pct=0.015, fallback_stop_loss_pct=0.01
        )
        plan.legs[0].exit_order_exchange_id = "201"
        plan.legs[0].exit_client_order_id = "tx201"
        plan.legs[1].exit_order_exchange_id = "202"
        plan.legs[1].exit_client_order_id = "tx202"
        self._tiered_order(exit_plan=plan)
        # Resting TP orders are still open (NEW); price drops through the stop.
        self.adapter.orders["201"] = {"orderId": 201, "status": "NEW", "executedQty": "0"}
        self.adapter.orders["202"] = {"orderId": 202, "status": "NEW", "executedQty": "0"}
        self.adapter.bid_price = 98.0
        self.reconciler.reconcile({"BTCUSDT": snapshot("BTCUSDT", 98.0)})
        self.assertIn("201", self.adapter.canceled)
        self.assertIn("202", self.adapter.canceled)
        sells = [s for s in self.adapter.submitted if s["side"] == "SELL"]
        self.assertEqual(len(sells), 1)  # one market-ish sell of the remainder
        self.assertAlmostEqual(float(sells[0]["quantity"]), 1.0)


if __name__ == "__main__":
    unittest.main()
