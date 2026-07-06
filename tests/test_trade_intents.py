from __future__ import annotations

import unittest

from trading_agent.core.config import AppConfig
from trading_agent.core.models import Side, TradeIntent, TradeProposal
from trading_agent.core.risk import RiskGovernor, RuntimeState


def proposal() -> TradeProposal:
    return TradeProposal(
        symbol="BTCUSDT",
        side=Side.BUY,
        price=100.0,
        quantity=1.0,
        confidence=0.95,
        expected_edge_bps=30.0,
        risk_bps=100.0,
        stop_loss_pct=0.01,
        take_profit_pct=0.015,
        rationale="test",
        evidence_ids=["ev_1"],
    )


class TradeIntentTest(unittest.TestCase):
    def test_trade_intent_round_trips_to_trade_proposal(self) -> None:
        intent = TradeIntent.from_proposal(proposal(), source_agent="strategy")
        converted = intent.to_proposal()

        self.assertTrue(intent.id.startswith("ti_"))
        self.assertTrue(converted.id.startswith("tp_"))
        self.assertEqual(converted.symbol, intent.symbol)
        self.assertEqual(converted.side, intent.side)
        self.assertEqual(converted.price, intent.limit_price)
        self.assertEqual(converted.quantity, intent.quantity)
        self.assertEqual(converted.evidence_ids, intent.evidence_ids)

    def test_trade_intent_cannot_bypass_risk_governor(self) -> None:
        intent = TradeIntent(
            symbol="DOGEUSDT",
            side=Side.BUY,
            limit_price=100.0,
            quantity=1.0,
            confidence=0.99,
            expected_edge_bps=50.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.015,
            rationale="unsupported",
            evidence_ids=[],
        )

        decision = RiskGovernor().evaluate(
            intent.to_proposal(),
            [],
            RuntimeState(mode="testnet", open_position_count=0, kill_switch=False),
            AppConfig(),
        )

        self.assertFalse(decision.approved)
        self.assertIn("not in the allowlist", " ".join(decision.reasons))
        self.assertIn("no evidence", " ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
