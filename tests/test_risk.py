from __future__ import annotations

import unittest

from trading_agent.core.config import AppConfig, SizingConfig
from trading_agent.core.models import EvidenceRecord, Side, TradeProposal, utc_iso
from trading_agent.core.risk import RiskGovernor, RuntimeState, conviction_size


def evidence(symbol: str = "BTCUSDT") -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            agent="market_data_agent",
            source="test",
            symbol=symbol,
            kind="price_order_book",
            observed_at=utc_iso(),
            score=0.5,
            confidence=0.8,
            payload={},
        )
    ]


def proposal(symbol: str = "BTCUSDT", confidence: float = 0.8) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        side=Side.BUY,
        price=100.0,
        quantity=1.0,
        confidence=confidence,
        expected_edge_bps=30.0,
        risk_bps=100.0,
        stop_loss_pct=0.01,
        take_profit_pct=0.015,
        rationale="test",
        evidence_ids=["ev"],
    )


class RiskGovernorTest(unittest.TestCase):
    def test_rejects_unsupported_symbol(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal("DOGEUSDT"),
            evidence("DOGEUSDT"),
            RuntimeState(mode="testnet", open_position_count=0, kill_switch=False),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("not in the allowlist", " ".join(decision.reasons))

    def test_third_position_requires_ninety_percent_heuristic_score(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.89),
            evidence(),
            RuntimeState(mode="testnet", open_position_count=2, kill_switch=False),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("90% heuristic score", " ".join(decision.reasons))

    def test_blocks_fourth_open_position(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.95),
            evidence(),
            RuntimeState(mode="testnet", open_position_count=3, kill_switch=False),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("maximum open-position cap reached", decision.reasons)

    def test_live_mode_requires_enablement_and_venue_confirmation(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.95),
            evidence(),
            RuntimeState(mode="live", open_position_count=0, kill_switch=False),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("live trading is disabled", decision.reasons)
        self.assertIn("live venue/account constraints are not confirmed", decision.reasons)

    def test_approves_valid_testnet_proposal_with_available_quote_balance(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(
                mode="testnet",
                open_position_count=0,
                kill_switch=False,
                available_quote_balance_usd=1_000.0,
            ),
            config,
        )
        self.assertTrue(decision.approved)

    def test_rejects_when_proposal_exceeds_available_quote_balance(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(
                mode="testnet",
                open_position_count=0,
                kill_switch=False,
                available_quote_balance_usd=50.0,
            ),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("proposal notional exceeds available quote balance", decision.reasons)

    def test_exchange_budget_can_scale_past_static_cap_when_balance_is_available(self) -> None:
        config = AppConfig()
        config.risk.max_open_positions = 200
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(
                mode="testnet",
                open_position_count=10,
                kill_switch=False,
                open_notional_usd=950.0,
                available_quote_balance_usd=200.0,
            ),
            config,
        )
        self.assertTrue(decision.approved)

    def test_kill_switch_blocks_approval(self) -> None:
        config = AppConfig()
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(mode="testnet", open_position_count=0, kill_switch=True),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("kill switch is enabled", decision.reasons)

    def test_per_trade_risk_prices_in_slippage_and_gap_buffer(self) -> None:
        # Notional 100, stop 1% -> base risk 1.0; default budget*fraction = 10.
        # A large gap buffer makes the worst-case loss exceed the per-trade cap
        # even though the bare stop distance would pass, proving the buffer is
        # actually priced into the check.
        config = AppConfig()
        config.risk.stop_gap_buffer_pct = 0.20
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(mode="testnet", open_position_count=0, kill_switch=False),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("per-trade risk exceeds configured risk fraction", decision.reasons)

    def test_correlated_exposure_cap_blocks_stacked_majors(self) -> None:
        config = AppConfig()  # max_correlated_notional_usd default 200
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),  # notional 100
            evidence(),
            RuntimeState(
                mode="testnet",
                open_position_count=1,
                kill_switch=False,
                available_quote_balance_usd=10_000.0,
                correlated_open_notional_usd=180.0,
            ),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("aggregate correlated (major) exposure", " ".join(decision.reasons))

    def test_daily_loss_circuit_breaker_halts_new_entries(self) -> None:
        config = AppConfig()  # daily_loss_halt_pct 0.05 * budget 100 = 5.0 limit
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(
                mode="testnet",
                open_position_count=0,
                kill_switch=False,
                available_quote_balance_usd=10_000.0,
                realized_pnl_today=-6.0,
            ),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("daily loss limit reached", " ".join(decision.reasons))

    def test_rejects_when_aggregate_notional_exceeds_budget(self) -> None:
        config = AppConfig()  # budget falls back to live capital budget when no balance is known
        decision = RiskGovernor().evaluate(
            proposal(confidence=0.8),
            evidence(),
            RuntimeState(
                mode="testnet", open_position_count=1, kill_switch=False, open_notional_usd=950.0
            ),
            config,
        )
        self.assertFalse(decision.approved)
        self.assertIn("aggregate open notional", " ".join(decision.reasons))


class ConvictionSizeTest(unittest.TestCase):
    def _size(self, **kw: float) -> float:
        params = dict(
            confidence=0.70,
            expected_edge_bps=40.0,
            atr_pct=None,
            budget_usd=100.0,
            sizing=SizingConfig(),
        )
        params.update(kw)
        return conviction_size(**params)  # type: ignore[arg-type]

    def test_monotonic_in_confidence(self) -> None:
        low = self._size(confidence=0.55)
        mid = self._size(confidence=0.70)
        high = self._size(confidence=0.85)
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_monotonic_in_edge(self) -> None:
        self.assertLess(self._size(expected_edge_bps=30.0), self._size(expected_edge_bps=60.0))

    def test_clamped_to_max_notional_and_budget(self) -> None:
        # Full conviction + full edge saturates at kelly_fraction*budget, capped.
        full = self._size(confidence=0.99, expected_edge_bps=120.0)
        self.assertLessEqual(full, 100.0)
        # A budget smaller than the computed size clamps to the budget.
        capped = self._size(confidence=0.99, expected_edge_bps=120.0, budget_usd=40.0)
        self.assertLessEqual(capped, 40.0)

    def test_high_volatility_shrinks_size(self) -> None:
        calm = self._size(atr_pct=0.005)  # below vol_target -> no trim
        wild = self._size(atr_pct=0.05)  # well above vol_target -> trimmed
        self.assertLess(wild, calm)

    def test_regime_and_quality_only_shrink(self) -> None:
        base = self._size()
        self.assertLess(self._size(regime_mult=0.5), base)
        self.assertLess(self._size(quality_mult=0.5), base)

    def test_sub_min_notional_returns_zero(self) -> None:
        # Edge at zero -> floor size only; a tiny budget falls below min_notional.
        self.assertEqual(self._size(expected_edge_bps=0.0, budget_usd=20.0), 0.0)

    def test_below_floor_confidence_yields_floor_size_not_zero(self) -> None:
        # At the conviction floor the conviction factor is 0 but a valid edge still
        # produces the minimum participation size (the anti-freeze guarantee).
        sized = self._size(confidence=0.50, expected_edge_bps=40.0, budget_usd=100.0)
        self.assertGreaterEqual(sized, SizingConfig().min_notional_usd)


class ConfidenceFloorByModeTest(unittest.TestCase):
    def test_testnet_uses_looser_floor_than_live(self) -> None:
        config = AppConfig()
        # 0.45 is below the live floor (0.50) but above the testnet floor (0.40).
        testnet = RiskGovernor().evaluate(
            proposal(confidence=0.45),
            evidence(),
            RuntimeState(
                mode="testnet",
                open_position_count=0,
                kill_switch=False,
                available_quote_balance_usd=1_000.0,
            ),
            config,
        )
        self.assertNotIn("proposal confidence is below minimum", testnet.reasons)
        live = RiskGovernor().evaluate(
            proposal(confidence=0.45),
            evidence(),
            RuntimeState(mode="live", open_position_count=0, kill_switch=False),
            config,
        )
        self.assertIn("proposal confidence is below minimum", live.reasons)


if __name__ == "__main__":
    unittest.main()
