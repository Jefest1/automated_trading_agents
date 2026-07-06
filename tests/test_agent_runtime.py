from __future__ import annotations

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trading_agent.cli import main
from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.models import Side, TradeIntent, TradeProposal
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime


class SupervisorRuntimeTest(unittest.TestCase):
    def test_agent_once_persists_checkpoint_heartbeat_prompts_and_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=Settings())
                result = runtime.run_once(cycle=1)
                summary = store.summary()
                subagents = runtime.subagent_specs()

        self.assertEqual(result.cycle, 1)
        self.assertEqual(summary["latest_agent_run"]["status"], "completed")
        self.assertEqual(summary["heartbeat"]["status"], "completed")
        self.assertEqual(summary["agent_checkpoint"]["last_cycle"], 1)
        self.assertGreaterEqual(summary["prompt_logs_count"], 7)
        self.assertTrue(
            any("/skills/binance-readonly" in agent["skills"] for agent in subagents if agent["name"] == "market_research")
        )
        self.assertTrue(any(agent["tools"] for agent in subagents if agent["name"] == "market_research"))
        self.assertFalse(
            any(".agents/skills" in skill for agent in subagents for skill in agent["skills"])
        )

    def test_kill_switch_blocks_trade_intent_execution(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                store.set_setting("kill_switch", True)
                result = SupervisorRuntime(config, store, settings=Settings()).run_once(cycle=1)
                orders = store.all_orders()

        self.assertGreater(result.intent_count, 0)
        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(orders, [])

    def test_runtime_requires_api_key_when_llm_supervisor_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is required"):
                    SupervisorRuntime(
                        config,
                        store,
                        settings=Settings(TRADING_AGENT_ENABLE_LLM_SUPERVISOR="true"),
                    )

    def test_testnet_execution_requires_explicit_enable_flag(self) -> None:
        proposal = _proposal()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            settings = Settings(
                TRADING_AGENT_EXECUTION_MODE="testnet",
                BINANCE_VENUE="testnet",
                BINANCE_API_BASE_URL="https://testnet.binance.vision/api",
                BINANCE_API_KEY="key",
                BINANCE_API_SECRET="secret",
            )
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                with self.assertRaisesRegex(RuntimeError, "TRADING_AGENT_ENABLE_TESTNET_ORDERS"):
                    runtime._submit_testnet_limit_entry(
                        proposal,
                        TradeIntent.from_proposal(proposal),
                        "run_test",
                    )

    def test_testnet_execution_uses_adapter_boundary(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeAdapter:
            def __init__(self, base_url: str, settings: Settings) -> None:
                self.base_url = base_url
                self.settings = settings

            @staticmethod
            def credentials_from_env(settings: Settings) -> object:
                return object()

            @staticmethod
            def quantize_order(symbol: str, quantity: float, price: float) -> tuple[str, str]:
                return str(quantity), str(price)

            def submit_limit_order(
                self,
                credentials: object,
                symbol: str,
                side: str,
                quantity: float,
                price: float,
                *,
                client_order_id: str | None = None,
            ) -> dict[str, object]:
                calls.append(
                    {
                        "base_url": self.base_url,
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "client_order_id": client_order_id,
                    }
                )
                return {"raw": {"orderId": 123, "status": "NEW"}}

        proposal = _proposal()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            settings = Settings(
                TRADING_AGENT_EXECUTION_MODE="testnet",
                TRADING_AGENT_ENABLE_TESTNET_ORDERS="true",
                BINANCE_VENUE="testnet",
                BINANCE_API_BASE_URL="https://testnet.binance.vision/api",
                BINANCE_API_KEY="key",
                BINANCE_API_SECRET="secret",
            )
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                with patch("trading_agent.graph.runtime.BinanceSpotAdapter", FakeAdapter):
                    order = runtime._submit_testnet_limit_entry(
                        proposal,
                        TradeIntent.from_proposal(proposal),
                        "run_test",
                    )

        self.assertEqual(order.mode, "testnet")
        self.assertEqual(order.order_type, "SPOT_TESTNET_LIMIT_ENTRY")
        self.assertEqual(calls[0]["base_url"], "https://testnet.binance.vision/api")
        self.assertEqual(calls[0]["symbol"], "BTCUSDT")
        self.assertEqual(calls[0]["side"], "BUY")

    def test_daemon_lock_prevents_duplicate_agent_runner(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                self.assertTrue(store.try_acquire_agent_lock("existing-owner"))

            with redirect_stdout(StringIO()):
                code = main(
                    [
                        "--home",
                        str(root),
                        "agent",
                        "run",
                        "--max-cycles",
                        "1",
                        "--interval-seconds",
                        "0",
                    ]
                )

        self.assertEqual(code, 1)

    def test_agent_run_duration_zero_exits_without_cycle(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
                code = main(
                    [
                        "--home",
                        str(root),
                        "agent",
                        "run",
                        "--duration-hours",
                        "0",
                        "--interval-seconds",
                        "0",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["duration_seconds"], 0.0)
        self.assertIsNone(payload["summary"]["latest_agent_run"])

    def test_agent_introduce_prints_agents_and_symbol_question(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
                code = main(["--home", str(root), "--env-file", ".env.test", "agent", "introduce"])

        payload = json.loads(output.getvalue())
        names = {agent["name"] for agent in payload["agents"]}
        self.assertEqual(code, 0)
        self.assertIn("supervisor", names)
        self.assertIn("market_research", names)
        self.assertNotIn("paper_execution", names)
        self.assertIn("Which tokens", payload["question"])
        self.assertEqual(payload["environment"]["trading_agent_home"], ".trading_agent_test")
        self.assertEqual(payload["source"], "deterministic")

    def test_agent_introduce_requires_api_key_when_llm_supervisor_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"TRADING_AGENT_HOME={root}",
                        "TRADING_AGENT_ENABLE_LLM_SUPERVISOR=true",
                        "OPENAI_API_KEY=",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is required"):
                main(["--env-file", str(env_file), "agent", "introduce"])

    def test_llm_introduction_uses_deep_agent_supervisor(self) -> None:
        class FakeDeepAgent:
            def invoke(self, payload, config):  # type: ignore[no-untyped-def]
                self.payload = payload
                self.config = config
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Deep Agent says: introduce agents and ask for BTCUSDT or ETHUSDT.",
                        }
                    ]
                }

            async def ainvoke(self, payload, config):  # type: ignore[no-untyped-def]
                return self.invoke(payload, config)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            fake_agent = FakeDeepAgent()
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(
                    config,
                    store,
                    settings=Settings(
                        TRADING_AGENT_ENABLE_LLM_SUPERVISOR="true",
                        OPENAI_API_KEY="test-key",
                    ),
                )
                runtime.build_deep_agent = lambda tools=None: fake_agent  # type: ignore[method-assign]
                payload = runtime.introduce(symbols=["BTCUSDT", "ETHUSDT"], thread_id="thread-test")

        self.assertEqual(payload["source"], "deep_agent")
        self.assertEqual(payload["message"], "Deep Agent says: introduce agents and ask for BTCUSDT or ETHUSDT.")
        self.assertIn("Introduce this trading-agent service", fake_agent.payload["messages"][0]["content"])
        self.assertEqual(fake_agent.config["configurable"]["thread_id"], "thread-test")

    def test_static_introduction_does_not_use_deep_agent_when_llm_disabled(self) -> None:
        config = AppConfig()
        settings = Settings(
            TRADING_AGENT_ENABLE_LLM_SUPERVISOR="false",
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=settings)
                payload = runtime.introduce()

        self.assertEqual(payload["source"], "deterministic")
        self.assertIn("agents", payload)

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


if __name__ == "__main__":
    unittest.main()
