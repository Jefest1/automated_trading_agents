from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.config import AppConfig, CostConfig, Settings, load_config, save_config
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime
from trading_agent.graph.cadence import FULL, REVIEW, SKIP, classify_cycle


class _Snap:
    def __init__(self, price: float) -> None:
        self.last_price = price


def _classify(**over):
    args = dict(
        baseline_intents=[],
        open_views=[],
        snapshots={"BTCUSDT": _Snap(100.0)},
        last_marks={"BTCUSDT": 100.0},
        minutes_since_review=999.0,
        is_first_cycle_of_day=False,
        cost=CostConfig(),
    )
    args.update(over)
    return classify_cycle(**args)


class ClassifyCycleTest(unittest.TestCase):
    def test_disabled_always_full(self) -> None:
        self.assertEqual(_classify(cost=CostConfig(enabled=False))[0], FULL)

    def test_baseline_entry_is_full(self) -> None:
        self.assertEqual(_classify(baseline_intents=[object()])[0], FULL)

    def test_first_cycle_of_day_is_full(self) -> None:
        self.assertEqual(_classify(is_first_cycle_of_day=True)[0], FULL)

    # Under the no-touch policy, bracket proximity / a moving PnL only warrants the
    # cheap news-sentry REVIEW (the exit ladder manages the bracket mechanically);
    # without a quiet_model it degrades to SKIP, never the expensive FULL desk.
    def test_position_near_take_profit_is_sentry_review(self) -> None:
        views = [{"symbol": "BTCUSDT", "to_take_profit_pct": 0.2, "to_stop_loss_pct": 5.0}]
        self.assertEqual(_classify(open_views=views)[0], REVIEW)
        self.assertEqual(_classify(open_views=views, cost=CostConfig(quiet_model=None))[0], SKIP)

    def test_position_near_stop_is_sentry_review(self) -> None:
        views = [{"symbol": "BTCUSDT", "to_take_profit_pct": 5.0, "to_stop_loss_pct": -0.2}]
        self.assertEqual(_classify(open_views=views)[0], REVIEW)

    def test_pnl_band_breach_is_sentry_review(self) -> None:
        views = [{"symbol": "BTCUSDT", "unrealized_pnl_pct": 1.2}]
        self.assertEqual(_classify(open_views=views)[0], REVIEW)

    def test_material_move_is_full(self) -> None:
        # +1% move vs last mark = 100 bps >= 50 bps default.
        tier, _ = _classify(snapshots={"BTCUSDT": _Snap(101.0)}, last_marks={"BTCUSDT": 100.0})
        self.assertEqual(tier, FULL)

    def test_held_quiet_position_is_review_when_quiet_model_set(self) -> None:
        views = [{"symbol": "BTCUSDT", "to_take_profit_pct": 5.0, "to_stop_loss_pct": 5.0, "unrealized_pnl_pct": 0.1}]
        cost = CostConfig(quiet_model="openai:gpt-5.1-mini")
        self.assertEqual(_classify(open_views=views, cost=cost)[0], REVIEW)

    def test_held_quiet_position_skips_without_quiet_model(self) -> None:
        views = [{"symbol": "BTCUSDT", "to_take_profit_pct": 5.0, "to_stop_loss_pct": 5.0, "unrealized_pnl_pct": 0.1}]
        self.assertEqual(_classify(open_views=views, cost=CostConfig(quiet_model=None))[0], SKIP)

    def test_held_quiet_position_reviews_with_quiet_model(self) -> None:
        views = [{"symbol": "BTCUSDT", "to_take_profit_pct": 5.0, "to_stop_loss_pct": 5.0, "unrealized_pnl_pct": 0.1}]
        # Default CostConfig now configures a mini quiet_model -> cheap REVIEW, not SKIP.
        self.assertEqual(_classify(open_views=views)[0], REVIEW)

    def test_baseline_entry_at_capacity_drops_to_review(self) -> None:
        # A new-entry signal with NO capacity must not force the expensive FULL cycle;
        # a held position falls through to the cheap REVIEW instead.
        views = [{"symbol": "ETHUSDT", "to_take_profit_pct": 5.0, "to_stop_loss_pct": 5.0, "unrealized_pnl_pct": 0.1}]
        tier, _ = _classify(baseline_intents=[object()], open_views=views, has_entry_capacity=False)
        self.assertEqual(tier, REVIEW)

    def test_flat_quiet_market_skips(self) -> None:
        self.assertEqual(_classify()[0], SKIP)


class CostConfigRoundTripTest(unittest.TestCase):
    def test_persists_and_merges(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "a.sqlite3"))
            config.cost.enabled = True
            config.cost.material_move_bps = 35.0
            config.cost.quiet_model = "openai:gpt-5.1-mini"
            config.risk.hard_maker_entry = False
            save_config(config)
            reloaded = load_config(root)
        self.assertAlmostEqual(reloaded.cost.material_move_bps, 35.0)
        self.assertEqual(reloaded.cost.quiet_model, "openai:gpt-5.1-mini")
        self.assertFalse(reloaded.risk.hard_maker_entry)


class SkipCycleGraphTest(unittest.TestCase):
    def test_skip_cycle_makes_no_deep_agent_call_and_waits(self) -> None:
        # Force SKIP: no open positions, no baseline proposal, not first-of-day.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            config.cost.full_on_first_cycle_of_day = False
            settings = Settings(TRADING_AGENT_ENABLE_LLM_SUPERVISOR="true", OPENAI_API_KEY="test-key")
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                # Deterministic baseline emits nothing -> no entry signal this cycle.
                runtime.strategy.propose = lambda *a, **k: []  # type: ignore[assignment]

                def _boom(*a, **k):
                    raise AssertionError("deep agent must not be built on a SKIP cycle")

                runtime.build_deep_agent = _boom  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                decisions = store.recent_supervisor_decisions()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.wait_count, 1)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(decisions[0]["action"], "WAIT")
        self.assertEqual(decisions[0]["source"], "deterministic")


if __name__ == "__main__":
    unittest.main()
