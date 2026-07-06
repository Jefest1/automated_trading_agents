from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.console import Console

from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.models import OrderRecord, OrderStatus, Side, utc_iso
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime
from trading_agent.repl.chat import handle_chat
from trading_agent.repl.renderer import AgentRenderer


class FakeDeepAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def ainvoke(self, payload, config):  # type: ignore[no-untyped-def]
        return {"messages": [{"role": "assistant", "content": self.reply}]}


def llm_settings() -> Settings:
    return Settings(TRADING_AGENT_ENABLE_LLM_SUPERVISOR="true", OPENAI_API_KEY="test-key")


def _install_fake_close(runtime: SupervisorRuntime) -> None:
    """Synchronous exchange close so the operator-chat CLOSE path completes
    without a live venue (paper execution removed)."""

    def _close(order, *, price=None, reason: str = "OPERATOR_CLOSE") -> OrderRecord:
        order.status = OrderStatus.CLOSED
        order.closed_at = utc_iso()
        order.closed_by = reason
        order.exit_price = price or order.price
        order.realized_pnl = round((order.exit_price - order.price) * order.quantity, 8)
        runtime.store.save_order(order)
        return order

    runtime.exchange_sync.close_position = _close  # type: ignore[method-assign]


def open_position(store: Store) -> OrderRecord:
    order = OrderRecord(
        proposal_id="tp_seed",
        mode="testnet",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type="SPOT_LIMIT_ENTRY",
        price=100.0,
        quantity=1.0,
        take_profit_price=1_000_000.0,
        stop_loss_price=0.000001,
        status=OrderStatus.POSITION_OPEN,
    )
    store.save_order(order)
    return order


def close_reply(order_id: str) -> str:
    decision = {
        "action": "CLOSE",
        "symbol": "BTCUSDT",
        "target_order_id": order_id,
        "rationale": "operator asked to close",
    }
    return "Closing as requested.\n```json\n" + json.dumps(decision) + "\n```"


def make_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, no_color=True, width=120)


class ChatFlowTest(unittest.TestCase):
    def test_chat_requires_llm_supervisor(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=Settings())
                with self.assertRaisesRegex(RuntimeError, "LLM_SUPERVISOR"):
                    runtime.chat("hi team")

    def test_chat_parses_decisions_without_executing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=llm_settings())
                _install_fake_close(runtime)
                order = open_position(store)
                runtime.build_deep_agent = lambda tools=None: FakeDeepAgent(close_reply(order.id))  # type: ignore[method-assign]
                result = runtime.chat("close my BTC position")
                still_open = [o.id for o in store.open_positions()]

        self.assertEqual(len(result["decisions"]), 1)
        self.assertEqual(result["decisions"][0].action.value, "CLOSE")
        self.assertIn(order.id, still_open)  # chat itself never executes

    def test_declined_confirmation_records_and_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=llm_settings())
                _install_fake_close(runtime)
                order = open_position(store)
                runtime.build_deep_agent = lambda tools=None: FakeDeepAgent(close_reply(order.id))  # type: ignore[method-assign]
                console = make_console()
                handle_chat(
                    runtime,
                    AgentRenderer(console),
                    console,
                    "close my BTC position",
                    confirm=lambda question: False,
                )
                still_open = [o.id for o in store.open_positions()]
                decisions = store.recent_supervisor_decisions()

        self.assertIn(order.id, still_open)
        declined = [d for d in decisions if d["gate_reasons"] == ["operator declined"]]
        self.assertEqual(len(declined), 1)

    def test_confirmed_close_executes_via_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=llm_settings())
                _install_fake_close(runtime)
                order = open_position(store)
                runtime.build_deep_agent = lambda tools=None: FakeDeepAgent(close_reply(order.id))  # type: ignore[method-assign]
                console = make_console()
                handle_chat(
                    runtime,
                    AgentRenderer(console),
                    console,
                    "close my BTC position",
                    confirm=lambda question: True,
                )
                closed = store.per_trade_pnl()

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["id"], order.id)
        self.assertEqual(closed[0]["closed_by"], "OPERATOR_CHAT_CLOSE")


if __name__ == "__main__":
    unittest.main()
