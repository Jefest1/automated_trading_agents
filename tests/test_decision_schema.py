from __future__ import annotations

import json
import unittest

from trading_agent.core.config import AppConfig
from trading_agent.core.decision import (
    DecisionAction,
    REQUIRED_AGENTS,
    decisions_from_intents,
    parse_supervisor_decisions,
    wait_decision,
)
from trading_agent.core.models import Side, TradeIntent


def consultations() -> list[dict]:
    return [
        {"agent": agent, "stance": "bullish", "confidence": 0.8, "summary": "ok"}
        for agent in sorted(REQUIRED_AGENTS)
    ]


def buy_block(**overrides) -> str:
    payload = {
        "action": "BUY",
        "symbol": "BTCUSDT",
        "limit_price": 105000.0,
        "quantity": 0.001,
        "confidence": 0.8,
        "expected_edge_bps": 20.0,
        "rationale": "test",
        "consultations": consultations(),
    }
    payload.update(overrides)
    return "Analysis...\n```json\n" + json.dumps(payload) + "\n```"


class ParseDecisionTest(unittest.TestCase):
    def test_valid_buy_decision_parses(self) -> None:
        decisions, errors = parse_supervisor_decisions(buy_block())
        self.assertEqual(errors, [])
        self.assertEqual(decisions[0].action, DecisionAction.BUY)
        self.assertEqual(decisions[0].symbol, "BTCUSDT")

    def test_buy_without_all_consultations_is_rejected(self) -> None:
        decisions, errors = parse_supervisor_decisions(buy_block(consultations=consultations()[:3]))
        self.assertEqual(decisions, [])
        self.assertTrue(any("consultations" in e for e in errors))

    def test_buy_without_price_is_rejected(self) -> None:
        decisions, errors = parse_supervisor_decisions(buy_block(limit_price=None))
        self.assertEqual(decisions, [])
        self.assertTrue(errors)

    def test_malformed_json_returns_errors(self) -> None:
        decisions, errors = parse_supervisor_decisions('```json\n{"action": "BUY", broken\n```')
        self.assertEqual(decisions, [])
        self.assertTrue(any("not valid JSON" in e for e in errors))

    def test_no_block_returns_error(self) -> None:
        decisions, errors = parse_supervisor_decisions("I think we should wait this cycle.")
        self.assertEqual(decisions, [])
        self.assertTrue(any("no decision JSON block" in e for e in errors))

    def test_wait_needs_only_action_symbol(self) -> None:
        text = '```json\n{"action": "WAIT", "symbol": "ETHUSDT", "rationale": "thin evidence"}\n```'
        decisions, errors = parse_supervisor_decisions(text)
        self.assertEqual(errors, [])
        self.assertEqual(decisions[0].action, DecisionAction.WAIT)

    def test_lowercase_wait_action_is_normalized(self) -> None:
        text = '```json\n{"action": "wait", "symbol": "ETHUSDT", "rationale": "thin evidence"}\n```'
        decisions, errors = parse_supervisor_decisions(text)
        self.assertEqual(errors, [])
        self.assertEqual(decisions[0].action, DecisionAction.WAIT)

    def test_consultation_stance_synonyms_are_normalized(self) -> None:
        rows = consultations()
        replacements = {
            "market_research": "slightly_bullish",
            "news_research": "sell",
            "onchain_research": "no_trade",
            "strategy": "buy",
            "risk_review": "prefer_wait",
            "reporting": "hold",
            "technical_analyst": "slightly_bullish",
        }
        for row in rows:
            row["stance"] = replacements[row["agent"]]
        decisions, errors = parse_supervisor_decisions(buy_block(consultations=rows))
        self.assertEqual(errors, [])
        stances = {row.agent: row.stance for row in decisions[0].consultations}
        self.assertEqual(stances["market_research"], "bullish")
        self.assertEqual(stances["news_research"], "bearish")
        self.assertEqual(stances["onchain_research"], "neutral")
        self.assertEqual(stances["strategy"], "bullish")
        self.assertEqual(stances["risk_review"], "neutral")
        self.assertEqual(stances["reporting"], "neutral")

    def test_approve_long_stance_does_not_reject_decision(self) -> None:
        # Regression: live cycle 7 (run_2da5a284baf84302) — risk_review returned
        # stance "approve_long" and the supervisor's only actionable BUY was
        # dropped by validation instead of being normalized.
        rows = consultations()
        for row in rows:
            if row["agent"] == "risk_review":
                row["stance"] = "approve_long"
            if row["agent"] == "strategy":
                row["stance"] = "reject_long"
        decisions, errors = parse_supervisor_decisions(buy_block(consultations=rows))
        self.assertEqual(errors, [])
        stances = {row.agent: row.stance for row in decisions[0].consultations}
        self.assertEqual(stances["risk_review"], "bullish")
        self.assertEqual(stances["strategy"], "bearish")

    def test_unknown_stance_coerces_to_abstain_instead_of_failing(self) -> None:
        rows = consultations()
        rows[0]["stance"] = "cosmic_alignment"
        rows[1]["stance"] = 42
        decisions, errors = parse_supervisor_decisions(buy_block(consultations=rows))
        self.assertEqual(errors, [])
        self.assertEqual(decisions[0].consultations[0].stance, "abstain")
        self.assertEqual(decisions[0].consultations[1].stance, "abstain")

    def test_inferred_directional_stances(self) -> None:
        rows = consultations()
        replacements = {
            "market_research": "strongly_bullish",
            "news_research": "approve short",
            "onchain_research": "avoid_long",
            "strategy": "REJECT-SHORT",
            "risk_review": "leaning bullish",
            "reporting": "no directional view",
            "technical_analyst": "strongly_bullish",
        }
        for row in rows:
            row["stance"] = replacements[row["agent"]]
        decisions, errors = parse_supervisor_decisions(buy_block(consultations=rows))
        self.assertEqual(errors, [])
        stances = {row.agent: row.stance for row in decisions[0].consultations}
        self.assertEqual(stances["market_research"], "bullish")
        self.assertEqual(stances["news_research"], "bearish")
        self.assertEqual(stances["onchain_research"], "bearish")
        self.assertEqual(stances["strategy"], "neutral")
        self.assertEqual(stances["risk_review"], "bullish")
        self.assertEqual(stances["reporting"], "abstain")

    def test_array_of_decisions(self) -> None:
        text = (
            "```json\n"
            '[{"action": "WAIT", "symbol": "BTCUSDT"},'
            ' {"action": "CLOSE", "symbol": "ETHUSDT", "target_order_id": "ord_1"}]\n'
            "```"
        )
        decisions, errors = parse_supervisor_decisions(text)
        self.assertEqual(errors, [])
        self.assertEqual([d.action for d in decisions], [DecisionAction.WAIT, DecisionAction.CLOSE])

    def test_close_requires_target_order(self) -> None:
        text = '```json\n{"action": "CLOSE", "symbol": "ETHUSDT"}\n```'
        decisions, errors = parse_supervisor_decisions(text)
        self.assertEqual(decisions, [])
        self.assertTrue(any("target_order_id" in e for e in errors))

    def test_adjust_requires_new_levels(self) -> None:
        text = '```json\n{"action": "ADJUST", "symbol": "ETHUSDT", "target_order_id": "ord_1"}\n```'
        decisions, errors = parse_supervisor_decisions(text)
        self.assertEqual(decisions, [])
        self.assertTrue(errors)

    def test_last_decision_block_wins(self) -> None:
        text = (
            '```json\n{"action": "WAIT", "symbol": "BTCUSDT"}\n```\n'
            "Revised after risk_review...\n"
            '```json\n{"action": "CLOSE", "symbol": "BTCUSDT", "target_order_id": "ord_9"}\n```'
        )
        decisions, errors = parse_supervisor_decisions(text)
        self.assertEqual(errors, [])
        self.assertEqual(decisions[0].action, DecisionAction.CLOSE)


class DeterministicFallbackTest(unittest.TestCase):
    def test_intents_become_buy_decisions_with_synthetic_consultations(self) -> None:
        intent = TradeIntent(
            symbol="BTCUSDT",
            side=Side.BUY,
            limit_price=100.0,
            quantity=1.0,
            confidence=0.8,
            expected_edge_bps=15.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.015,
            rationale="baseline",
            evidence_ids=["ev_1"],
        )
        decisions = decisions_from_intents([intent])
        self.assertEqual(decisions[0].action, DecisionAction.BUY)
        self.assertEqual(decisions[0].source, "deterministic")
        self.assertEqual({c.agent for c in decisions[0].consultations}, set(REQUIRED_AGENTS))
        round_trip = decisions[0].to_intent(AppConfig())
        self.assertEqual(round_trip.symbol, "BTCUSDT")
        self.assertEqual(round_trip.limit_price, 100.0)

    def test_wait_decision_helper(self) -> None:
        decision = wait_decision("BTCUSDT", "because")
        self.assertEqual(decision.action, DecisionAction.WAIT)
        self.assertEqual(decision.rationale, "because")


if __name__ == "__main__":
    unittest.main()
