from __future__ import annotations

import unittest

from trading_agent.agents.strategy import StrategyAgent
from trading_agent.core.config import AppConfig
from trading_agent.core.decision import (
    REQUIRED_AGENTS,
    AgentConsultation,
    DecisionAction,
    SupervisorDecision,
    decisions_from_intents,
)
from trading_agent.core.exit_ladder import build_exit_plan
from trading_agent.core.models import (
    EvidenceRecord,
    LevelMap,
    MarketSnapshot,
    Side,
    SupportZone,
    TradeIntent,
    utc_iso,
)


def _snapshot(symbol: str = "BTCUSDT", price: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        observed_at=utc_iso(),
        last_price=price,
        bid_price=price - 0.05,
        ask_price=price + 0.05,
        volume_24h=1_000_000.0,
        atr=1.0,
    )


def _evidence(symbol: str = "BTCUSDT", score: float = 0.4) -> EvidenceRecord:
    return EvidenceRecord(
        agent="market_data_agent",
        source="binance-compatible-feed",
        symbol=symbol,
        kind="price_order_book",
        observed_at=utc_iso(),
        score=score,
        confidence=0.7,
        payload={},
    )


def _support(low: float, high: float, strength: float, price: float) -> SupportZone:
    mid = (low + high) / 2
    return SupportZone(
        low=low,
        high=high,
        mid=mid,
        strength=strength,
        side="support",
        methods=["hvn", "swing_low"],
        timeframes=["1d"],
        distance_pct=(mid - price) / price,
        touches=3,
    )


def _resistance(low: float, high: float, price: float) -> SupportZone:
    mid = (low + high) / 2
    return SupportZone(
        low=low,
        high=high,
        mid=mid,
        strength=4.0,
        side="resistance",
        methods=["swing_high"],
        timeframes=["1d"],
        distance_pct=(mid - price) / price,
        touches=2,
    )


def _level_map(regime: str, price: float = 100.0) -> LevelMap:
    return LevelMap(
        symbol="BTCUSDT",
        current_price=price,
        regime=regime,
        support_zones=[
            _support(96.5, 97.0, 6.0, price),  # ~3% down
            _support(93.5, 94.0, 5.0, price),  # ~6% down
        ],
        resistance_zones=[_resistance(108.0, 109.0, price)],
    )


class ZoneLadderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = StrategyAgent()
        self.config = AppConfig()

    def test_uptrend_produces_laddered_zone_bids_with_stops_below_zones(self) -> None:
        snaps = {"BTCUSDT": _snapshot()}
        proposals = self.agent.propose(
            snaps, [_evidence()], self.config, {"BTCUSDT": _level_map("uptrend")}
        )
        self.assertGreaterEqual(len(proposals), 2)  # laddered
        ladder_ids = {p.ladder_id for p in proposals}
        self.assertEqual(len(ladder_ids), 1)  # one shared ladder
        for p in proposals:
            self.assertIsNotNone(p.zone_id)
            self.assertIsNotNone(p.stop_price)
            self.assertIsNotNone(p.target_price)
            self.assertLess(p.stop_price, p.price)  # stop below entry
            self.assertGreater(p.target_price, p.price)  # target above entry
            self.assertLess(p.price, snaps["BTCUSDT"].last_price)  # bid below market
            self.assertIsNotNone(p.expires_at)  # TTL stamped
            # geometric edge reflects the real move to target
            self.assertGreater(p.expected_edge_bps, self.config.risk.min_expected_edge_bps)

    def test_downtrend_bids_only_one_deepest_zone(self) -> None:
        snaps = {"BTCUSDT": _snapshot()}
        proposals = self.agent.propose(
            snaps, [_evidence()], self.config, {"BTCUSDT": _level_map("downtrend")}
        )
        self.assertLessEqual(len(proposals), 1)

    def test_bearish_read_does_not_bid_support(self) -> None:
        snaps = {"BTCUSDT": _snapshot()}
        proposals = self.agent.propose(
            snaps, [_evidence(score=-0.5)], self.config, {"BTCUSDT": _level_map("uptrend")}
        )
        # Clearly bearish evidence: no demand-zone accumulation.
        self.assertEqual([p for p in proposals if p.zone_id], [])

    def test_no_level_map_falls_back_to_legacy_maker_entry(self) -> None:
        snaps = {"BTCUSDT": _snapshot()}
        proposals = self.agent.propose(snaps, [_evidence(score=0.6)], self.config, None)
        # Legacy path: a proposal with no zone anchoring (or none if edge too thin).
        for p in proposals:
            self.assertIsNone(p.zone_id)


class BookedStopMatchesZoneTest(unittest.TestCase):
    """Regression: the booked exit bracket must use the zone stop the desk reasoned,
    not the config 4% tiered stop. (Live bug: BTC/BNB/ETH all booked at 4%.)"""

    def setUp(self) -> None:
        self.config = AppConfig()  # exits.enabled True (tiered) by default
        self.assertTrue(self.config.exits.enabled)

    @staticmethod
    def _consults() -> list[AgentConsultation]:
        return [
            AgentConsultation(agent=a, stance="bullish", confidence=0.7) for a in sorted(REQUIRED_AGENTS)
        ]

    def _booked_stop(self, decision: SupervisorDecision) -> float:
        intent = decision.to_intent(self.config)
        proposal = intent.to_proposal()
        plan = build_exit_plan(
            proposal.price,
            self.config.exits,
            fallback_take_profit_pct=proposal.take_profit_pct,
            fallback_stop_loss_pct=proposal.stop_loss_pct,
            stop_price=proposal.stop_price,
        )
        return plan.current_stop_price

    def test_deterministic_zone_stop_is_booked_not_config_4pct(self) -> None:
        # A deterministic zone bid: entry 63500.9, zone stop 62515.66 (~1.55%).
        intent = TradeIntent(
            symbol="BTCUSDT",
            side=Side.BUY,
            limit_price=63500.9,
            quantity=0.0007,
            confidence=0.64,
            expected_edge_bps=600.0,
            stop_loss_pct=(63500.9 - 62515.66) / 63500.9,
            take_profit_pct=(67310.0 / 63500.9) - 1,
            rationale="zone bid",
            evidence_ids=["ev_x"],
            target_price=67310.0,
            stop_price=62515.66,
        )
        decision = decisions_from_intents([intent])[0]
        booked = self._booked_stop(decision)
        self.assertAlmostEqual(booked, 62515.66, places=2)
        # NOT the config 4% stop (which would be ~60960.86).
        self.assertGreater(booked, 63500.9 * (1 - self.config.exits.initial_stop_loss_pct))

    def test_llm_absolute_stop_price_is_booked(self) -> None:
        decision = SupervisorDecision(
            action=DecisionAction.BUY,
            symbol="SOLUSDT",
            limit_price=72.58,
            quantity=1.0,
            stop_price=72.08,  # absolute invalidation the LLM reasoned
            target_price=73.84,
            confidence=0.76,
            consultations=self._consults(),
        )
        self.assertAlmostEqual(self._booked_stop(decision), 72.08, places=4)

    def test_pct_only_decision_still_derives_absolute_stop(self) -> None:
        decision = SupervisorDecision(
            action=DecisionAction.BUY,
            symbol="SOLUSDT",
            limit_price=100.0,
            quantity=1.0,
            stop_loss_pct=0.01,  # 1% -> stop 99.0, must be booked (not config 4%)
            take_profit_pct=0.03,
            confidence=0.7,
            consultations=self._consults(),
        )
        self.assertAlmostEqual(self._booked_stop(decision), 99.0, places=4)


if __name__ == "__main__":
    unittest.main()
