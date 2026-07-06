from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.config import AppConfig
from trading_agent.core.models import OrderRecord, OrderStatus, RiskDecision, Side, utc_iso
from trading_agent.core.reporting import promotion_report, report_markdown
from trading_agent.core.storage import Store, is_risk_breach


def closed_trade(symbol: str = "BNBUSDT", pnl: float = -1.2) -> OrderRecord:
    return OrderRecord(
        proposal_id="tp_1",
        mode="testnet",
        symbol=symbol,
        side=Side.BUY,
        order_type="SPOT_LIMIT_ENTRY",
        price=700.0,
        quantity=0.14,
        take_profit_price=710.0,
        stop_loss_price=693.0,
        status=OrderStatus.CLOSED,
        closed_at=utc_iso(),
        exit_price=693.0,
        exit_reason="STOP_LOSS",
        realized_pnl=pnl,
    )


def open_position(symbol: str = "SOLUSDT") -> OrderRecord:
    return OrderRecord(
        proposal_id="tp_2",
        mode="testnet",
        symbol=symbol,
        side=Side.BUY,
        order_type="SPOT_LIMIT_ENTRY",
        price=178.0,
        quantity=0.56,
        take_profit_price=181.0,
        stop_loss_price=176.0,
        status=OrderStatus.POSITION_OPEN,
    )


class ReportingTest(unittest.TestCase):
    def test_promotion_report_requires_positive_pnl_and_no_breaches(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                report = promotion_report(store, config)

        self.assertFalse(report["promotion_gate"]["positive_net_pnl_after_costs"])
        self.assertTrue(report["promotion_gate"]["zero_risk_control_breaches"])
        self.assertFalse(report["promotion_gate"]["eligible_for_live_capital_test"])

    def test_report_breaks_pnl_down_per_trade_and_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                store.save_order(closed_trade())
                store.save_order(open_position())
                report = promotion_report(store, config, prices={"SOLUSDT": 180.0})

        self.assertEqual(report["pnl"]["realized_total"], -1.2)
        self.assertEqual(report["pnl"]["realized_by_mode"], {"testnet": -1.2})
        self.assertEqual(len(report["pnl"]["closed_trades"]), 1)
        position = report["open_positions"][0]
        self.assertEqual(position["current_price"], 180.0)
        self.assertAlmostEqual(position["unrealized_pnl"], (180.0 - 178.0) * 0.56)
        self.assertAlmostEqual(report["pnl"]["unrealized_open_total"], (180.0 - 178.0) * 0.56)

    def test_unpriced_open_position_marks_unrealized_unknown(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                store.save_order(open_position())
                report = promotion_report(store, config)

        self.assertIsNone(report["pnl"]["unrealized_open_total"])
        self.assertIsNone(report["open_positions"][0]["unrealized_pnl"])


class BreachVsVetoTest(unittest.TestCase):
    def test_confidence_veto_is_not_a_breach(self) -> None:
        self.assertFalse(is_risk_breach(["proposal confidence is below minimum"]))
        self.assertFalse(is_risk_breach(["expected edge is below minimum"]))
        self.assertFalse(is_risk_breach(["maximum open-position cap reached"]))

    def test_kill_switch_and_daily_loss_are_breaches(self) -> None:
        self.assertTrue(is_risk_breach(["kill switch is enabled"]))
        self.assertTrue(is_risk_breach(["daily loss limit reached; new entries halted"]))

    def test_counts_split_vetoes_from_breaches(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                # Two routine confidence/edge vetoes + one genuine breach.
                store.save_risk_decision(
                    RiskDecision(proposal_id="tp_a", approved=False, reasons=["proposal confidence is below minimum"])
                )
                store.save_risk_decision(
                    RiskDecision(proposal_id="tp_b", approved=False, reasons=["expected edge is below minimum"])
                )
                store.save_risk_decision(
                    RiskDecision(proposal_id="tp_c", approved=False, reasons=["kill switch is enabled"])
                )
                self.assertEqual(store.count_vetoes(), 3)
                self.assertEqual(store.count_risk_breaches(), 1)
                report = promotion_report(store, config)
        # The two confidence/edge vetoes must NOT block the breach gate.
        self.assertTrue(report["promotion_gate"]["zero_risk_control_breaches"] is False)
        self.assertEqual(report["promotion_gate"]["vetoes"], 3)

    def test_markdown_includes_per_trade_table_and_rejection_label(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                store.save_order(closed_trade())
                store.save_order(open_position())
                output = report_markdown(store, config, prices={"SOLUSDT": 180.0})

        self.assertIn("## PnL (per trade, after fees)", output)
        self.assertIn("### Closed trades", output)
        self.assertIn("STOP_LOSS", output)
        self.assertIn("### Open positions", output)
        # Vetoes (healthy gate rejections) and real breaches are reported separately.
        self.assertIn("Gate vetoes (healthy, marginal trades declined)", output)
        self.assertIn("Risk-control breaches (alarming)", output)


if __name__ == "__main__":
    unittest.main()
