from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.models import Side, TradeIntent, TradeProposal
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime


def _proposal() -> TradeProposal:
    return TradeProposal(
        symbol="BTCUSDT",
        side=Side.BUY,
        price=90000.0,
        quantity=0.001,
        confidence=0.95,
        expected_edge_bps=25.0,
        risk_bps=100.0,
        stop_loss_pct=0.01,
        take_profit_pct=0.015,
        rationale="test proposal",
        evidence_ids=["ev_1"],
    )


class ExchangeBaseUrlTest(unittest.TestCase):
    def test_testnet_mode_pins_testnet_url(self) -> None:
        settings = Settings(TRADING_AGENT_EXECUTION_MODE="testnet", BINANCE_VENUE="testnet")
        self.assertEqual(settings.exchange_base_url(), "https://testnet.binance.vision/api")

    def test_live_binance_com_resolves_production_url(self) -> None:
        settings = Settings(TRADING_AGENT_EXECUTION_MODE="live", BINANCE_VENUE="binance.com")
        self.assertEqual(settings.exchange_base_url(), "https://api.binance.com/api")

    def test_live_binance_us_resolves_us_url(self) -> None:
        settings = Settings(TRADING_AGENT_EXECUTION_MODE="live", BINANCE_VENUE="binance.us")
        self.assertEqual(settings.exchange_base_url(), "https://api.binance.us/api")

    def test_live_mode_is_a_valid_execution_mode(self) -> None:
        settings = Settings(TRADING_AGENT_EXECUTION_MODE="live")
        self.assertEqual(settings.trading_agent_execution_mode, "live")


class LiveGatingTest(unittest.TestCase):
    def test_live_order_blockers_list_all_missing_flags(self) -> None:
        settings = Settings(TRADING_AGENT_EXECUTION_MODE="live")
        blockers = settings.live_order_blockers()
        self.assertTrue(any("ENABLE_LIVE_ORDERS" in b for b in blockers))
        self.assertTrue(any("BINANCE_VENUE" in b for b in blockers))
        self.assertTrue(any("BINANCE_API_KEY" in b for b in blockers))

    def test_live_submission_requires_every_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            settings = Settings(
                TRADING_AGENT_EXECUTION_MODE="live",
                BINANCE_VENUE="binance.com",
                BINANCE_API_KEY="key",
                BINANCE_API_SECRET="secret",
                # TRADING_AGENT_ENABLE_LIVE_ORDERS intentionally missing
            )
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                proposal = _proposal()
                with self.assertRaisesRegex(RuntimeError, "TRADING_AGENT_ENABLE_LIVE_ORDERS"):
                    runtime._submit_exchange_limit_entry(
                        proposal, TradeIntent.from_proposal(proposal), "run_test"
                    )

    def test_live_submission_requires_config_json_confirmation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            settings = Settings(
                TRADING_AGENT_EXECUTION_MODE="live",
                TRADING_AGENT_ENABLE_LIVE_ORDERS="true",
                BINANCE_VENUE="binance.com",
                BINANCE_API_KEY="key",
                BINANCE_API_SECRET="secret",
            )
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                proposal = _proposal()
                with self.assertRaisesRegex(RuntimeError, "live.enabled"):
                    runtime._submit_exchange_limit_entry(
                        proposal, TradeIntent.from_proposal(proposal), "run_test"
                    )

    def test_autonomous_live_orders_blocked_without_caps_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            config.live.enabled = True
            config.live.venue_confirmed = True
            config.live.auto_orders_within_caps = False
            settings = Settings(
                TRADING_AGENT_EXECUTION_MODE="live",
                TRADING_AGENT_ENABLE_LIVE_ORDERS="true",
                BINANCE_VENUE="binance.com",
                BINANCE_API_KEY="key",
                BINANCE_API_SECRET="secret",
            )
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                proposal = _proposal()
                with self.assertRaisesRegex(RuntimeError, "autonomous live orders are disabled"):
                    runtime._submit_exchange_limit_entry(
                        proposal, TradeIntent.from_proposal(proposal), "run_test"
                    )


class SubagentModelOverridesTest(unittest.TestCase):
    def test_config_and_env_overrides_apply_to_known_agents_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            config.model.subagent_models = {
                "news_research": "openai:gpt-5.1-mini",
                "made_up_agent": "openai:gpt-5.1",
            }
            settings = Settings(
                TRADING_AGENT_SUBAGENT_MODELS='{"strategy": "anthropic:claude-sonnet-4-6"}'
            )
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                # Assert on the validated override map (config + env merge, known
                # agents only). subagent_specs() turns these into model INSTANCES,
                # which would require live provider keys to build.
                models = runtime._subagent_models()

        self.assertEqual(models["news_research"], "openai:gpt-5.1-mini")
        self.assertEqual(models["strategy"], "anthropic:claude-sonnet-4-6")
        self.assertNotIn("made_up_agent", models)  # unknown agent dropped
        self.assertNotIn("market_research", models)  # no override -> inherits supervisor


class DebateTierGatingTest(unittest.TestCase):
    def test_full_includes_debate_review_excludes_it(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=Settings())
                full = {spec["name"] for spec in runtime.subagent_specs()}
                review = {
                    spec["name"]
                    for spec in runtime.subagent_specs(only=config.cost.review_subagents)
                }
        # FULL cycles run the adversarial debate; cheap REVIEW cycles do not.
        self.assertIn("bull_researcher", full)
        self.assertIn("bear_researcher", full)
        self.assertNotIn("bull_researcher", review)
        self.assertNotIn("bear_researcher", review)


if __name__ == "__main__":
    unittest.main()
