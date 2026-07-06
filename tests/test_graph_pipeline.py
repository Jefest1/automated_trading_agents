from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.decision import REQUIRED_AGENTS
from trading_agent.core.exit_ladder import build_exit_plan
from trading_agent.core.models import OrderRecord, OrderStatus, Side, utc_iso
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime


def install_fake_exchange(runtime: SupervisorRuntime) -> None:
    """Stub the testnet/live exchange so graph-pipeline tests exercise the full
    decision -> gate -> execute path without a real venue (paper execution removed)."""

    def _submit(proposal, intent, run_id, *, operator_initiated: bool = False) -> OrderRecord:
        plan = build_exit_plan(
            proposal.price,
            runtime.config.exits,
            fallback_take_profit_pct=proposal.take_profit_pct,
            fallback_stop_loss_pct=proposal.stop_loss_pct,
        )
        return OrderRecord(
            proposal_id=proposal.id,
            mode=runtime.settings.trading_agent_execution_mode,
            symbol=proposal.symbol,
            side=proposal.side,
            order_type="SPOT_TESTNET_LIMIT_ENTRY",
            price=proposal.price,
            quantity=proposal.quantity,
            take_profit_price=plan.legs[0].target_price,
            stop_loss_price=plan.current_stop_price,
            exit_plan=plan,
            status=OrderStatus.POSITION_OPEN,
            exchange_order_id="fake-exch-1",
        )

    def _close(order, *, price=None, reason: str = "OPERATOR_CLOSE") -> OrderRecord:
        order.status = OrderStatus.CLOSED
        order.closed_at = utc_iso()
        order.closed_by = reason
        order.exit_price = price or order.price
        order.realized_pnl = round((order.exit_price - order.price) * order.quantity, 8)
        runtime.store.save_order(order)
        return order

    runtime._submit_exchange_limit_entry = _submit  # type: ignore[method-assign]
    runtime.exchange_sync.close_position = _close  # type: ignore[method-assign]
    # Simulate a funded testnet quote balance so the gate sizes against real funds
    # instead of falling back to the small static capital budget.
    runtime.available_quote_balance = lambda symbol: 1_000_000.0  # type: ignore[method-assign]


class FakeDeepAgent:
    """Returns a canned supervisor reply containing a decision block."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.payloads: list[dict] = []

    def invoke(self, payload, config):  # type: ignore[no-untyped-def]
        self.payloads.append(payload)
        return {"messages": [{"role": "assistant", "content": self.reply}]}

    async def ainvoke(self, payload, config):  # type: ignore[no-untyped-def]
        return self.invoke(payload, config)


class StreamingOnlyFakeDeepAgent(FakeDeepAgent):
    """Streams a model update like Deep Agents did in the operator logs."""

    def stream(self, payload, *, config, stream_mode, subgraphs):  # type: ignore[no-untyped-def]
        yield (("consult_agents:abc",), "updates", {"model": {"messages": [{"content": self.reply}]}})

    async def astream(self, payload, *, config, stream_mode, subgraphs):  # type: ignore[no-untyped-def]
        yield (("consult_agents:abc",), "updates", {"model": {"messages": [{"content": self.reply}]}})

    def get_state(self, config):  # type: ignore[no-untyped-def]
        raise ValueError("No checkpointer set")

    async def aget_state(self, config):  # type: ignore[no-untyped-def]
        raise ValueError("No checkpointer set")


def consultations() -> list[dict]:
    return [
        {"agent": agent, "stance": "bullish", "confidence": 0.8, "summary": "ok"}
        for agent in sorted(REQUIRED_AGENTS)
    ]


def llm_settings() -> Settings:
    return Settings(TRADING_AGENT_ENABLE_LLM_SUPERVISOR="true", OPENAI_API_KEY="test-key")


class GraphPipelineTest(unittest.TestCase):
    def _runtime(self, root: Path, store: Store, settings: Settings) -> SupervisorRuntime:
        config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
        runtime = SupervisorRuntime(config, store, settings=settings)
        install_fake_exchange(runtime)
        return runtime

    def test_deterministic_mode_runs_full_pipeline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=Settings())
                result = runtime.run_once(cycle=1)
                decisions = store.recent_supervisor_decisions()

        self.assertGreater(result.decision_count, 0)
        # deterministic fallback records its decisions with source=deterministic
        self.assertTrue(all(d["source"] == "deterministic" for d in decisions))

    def test_llm_wait_decision_executes_nothing(self) -> None:
        reply = '```json\n{"action": "WAIT", "symbol": "BTCUSDT", "rationale": "thin evidence"}\n```'
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()
                decisions = store.recent_supervisor_decisions()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.wait_count, 1)
        self.assertEqual(orders, [])
        self.assertEqual(decisions[0]["action"], "WAIT")

    def test_streamed_llm_wait_decision_survives_missing_checkpoint_state(self) -> None:
        reply = '```json\n{"action": "WAIT", "symbol": "BTCUSDT", "rationale": "thin evidence"}\n```'
        events: list[tuple[str, str, dict]] = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                runtime.event_callback = lambda kind, agent, data: events.append((kind, agent, data))
                runtime.build_deep_agent = lambda tools=None, **_kw: StreamingOnlyFakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()
                decisions = store.recent_supervisor_decisions()

        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.wait_count, 1)
        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(orders, [])
        self.assertEqual(decisions[0]["action"], "WAIT")
        self.assertTrue(any(kind == "supervisor" for kind, _, _ in events))

    def test_llm_buy_decision_passes_gate_and_submits_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": snapshot.bid_price,
                    "quantity": round(100.0 / snapshot.bid_price, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "test buy",
                    "consultations": consultations(),
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()
                decisions = store.recent_supervisor_decisions()

        self.assertEqual(result.submitted_trades, 1)
        self.assertEqual(orders[0]["symbol"], "BTCUSDT")
        self.assertEqual(orders[0]["mode"], "testnet")
        executed = [d for d in decisions if d["executed_order_id"]]
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]["executed_order_id"], orders[0]["id"])

    def test_risk_review_size_multiplier_zero_vetoes_buy(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                consults = consultations()
                for consultation in consults:
                    if consultation["agent"] == "risk_review":
                        consultation["size_multiplier"] = 0.0  # binding veto
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": snapshot.bid_price,
                    "quantity": round(100.0 / snapshot.bid_price, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "risk_review should veto",
                    "consultations": consults,
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()
                decisions = store.recent_supervisor_decisions()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(orders, [])
        rejected = [d for d in decisions if not d["gate_approved"]]
        self.assertTrue(
            any("risk_review vetoed" in reason for r in rejected for reason in r["gate_reasons"])
        )

    def test_llm_buy_with_bad_price_is_vetoed_by_gate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                # With hard_maker_entry on, the gate would OVERRIDE a bad LLM price
                # with the maker-pullback price (so there is nothing to veto). Turn
                # it off so the absurd price reaches the >2% deviation pre-trade
                # blocker, which is the veto this test asserts. (Deterministic
                # conviction sizing now also overrides the LLM quantity, so the old
                # "absurd notional" veto path no longer applies.)
                runtime.config.risk.hard_maker_entry = False
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": 1.0,  # absurd price -> >2% deviation blocker
                    "quantity": 100.0,
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "bad price",
                    "consultations": consultations(),
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(orders, [])

    def test_llm_buy_with_fabricated_evidence_refs_is_rejected(self) -> None:
        # The supervisor cites evidence ids that were never gathered this cycle
        # (the chat.log failure mode). With require_evidence_refs on, the gate
        # must refuse instead of silently re-scoring on the symbol's evidence.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": snapshot.bid_price,
                    "quantity": round(100.0 / snapshot.bid_price, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "fabricated refs",
                    "consultations": consultations(),
                    "evidence_refs": ["market_BTCUSDT_price_order_book_2026-06-12"],
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()
                decisions = store.recent_supervisor_decisions()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(orders, [])
        rejected = [d for d in decisions if not d["gate_approved"]]
        self.assertTrue(
            any(
                "evidence_refs cite no evidence" in reason
                for r in rejected
                for reason in r["gate_reasons"]
            )
        )

    def test_fabricated_evidence_refs_allowed_when_enforcement_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            config.risk.require_evidence_refs = False
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=llm_settings())
                install_fake_exchange(runtime)
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": snapshot.bid_price,
                    "quantity": round(100.0 / snapshot.bid_price, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "fabricated refs tolerated",
                    "consultations": consultations(),
                    "evidence_refs": ["totally_made_up"],
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()

        self.assertEqual(result.submitted_trades, 1)
        self.assertEqual(len(orders), 1)

    def test_maker_pullback_buy_below_bid_submits(self) -> None:
        # A BUY resting below the bid is a maker-pullback entry; it should pass
        # the gate (the golden rule: rest below market, don't chase).
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                limit = round(snapshot.bid_price * 0.999, 2)  # ~10 bps below bid
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": limit,
                    "quantity": round(100.0 / limit, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "maker pullback",
                    "consultations": consultations(),
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()

        self.assertEqual(result.submitted_trades, 1)
        self.assertEqual(len(orders), 1)

    def test_buy_above_bid_is_refused_when_hard_maker_off(self) -> None:
        # Safety-net path: with the hard-maker override disabled, a BUY that chases
        # above the bid is still refused by the pre-trade cross check.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                runtime.config.risk.hard_maker_entry = False
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                limit = round(snapshot.ask_price * 1.005, 2)  # well above bid -> chase
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": limit,
                    "quantity": round(100.0 / limit, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "chasing",
                    "consultations": consultations(),
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(orders, [])

    def test_buy_above_bid_is_repriced_to_maker_by_default(self) -> None:
        # Hard maker discipline (default): a chasing BUY is REPRICED to a maker
        # limit at/below the bid and approved, instead of being refused.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                snapshot = runtime.feed.snapshot("BTCUSDT", 1)
                limit = round(snapshot.ask_price * 1.005, 2)  # chase
                decision = {
                    "action": "BUY",
                    "symbol": "BTCUSDT",
                    "limit_price": limit,
                    "quantity": round(100.0 / limit, 8),
                    "confidence": 0.95,
                    "expected_edge_bps": 35.0,
                    "rationale": "chasing",
                    "consultations": consultations(),
                }
                reply = "```json\n" + json.dumps(decision) + "\n```"
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                orders = store.all_orders()

        self.assertEqual(result.submitted_trades, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertLessEqual(orders[0]["price"], snapshot.bid_price)

    def test_llm_malformed_output_degrades_to_wait(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent("YOLO buy everything!!")  # type: ignore[method-assign]
                result = runtime.run_once(cycle=1, symbols=["BTCUSDT", "ETHUSDT"])
                orders = store.all_orders()

        self.assertEqual(result.submitted_trades, 0)
        self.assertEqual(result.wait_count, 2)
        self.assertEqual(orders, [])
        self.assertGreater(result.error_count, 0)

    def test_llm_close_decision_closes_open_position(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                position = OrderRecord(
                    proposal_id="tp_seed",
                    mode="testnet",
                    symbol="BTCUSDT",
                    side=Side.BUY,
                    order_type="SPOT_LIMIT_ENTRY",
                    price=100.0,
                    quantity=1.0,
                    take_profit_price=1_000_000.0,  # unreachable: reconcile keeps it open
                    stop_loss_price=0.000001,
                    status=OrderStatus.POSITION_OPEN,
                )
                store.save_order(position)
                runtime.config.risk.min_hold_hours = 0.0  # this test exercises the close path, not the cooldown
                reply = (
                    "```json\n"
                    + json.dumps(
                        {
                            "action": "CLOSE",
                            "symbol": "BTCUSDT",
                            "target_order_id": position.id,
                            "rationale": "lock in profit",
                        }
                    )
                    + "\n```"
                )
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                closed = store.per_trade_pnl()

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["id"], position.id)
        self.assertEqual(closed[0]["closed_by"], "SUPERVISOR_CLOSE")

    def test_min_hold_blocks_discretionary_close_of_fresh_position(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                runtime.config.risk.min_hold_hours = 4.0
                position = OrderRecord(
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
                )  # opened_at defaults to now -> inside the 4h cooldown
                store.save_order(position)
                reply = (
                    "```json\n"
                    + json.dumps(
                        {
                            "action": "CLOSE",
                            "symbol": "BTCUSDT",
                            "target_order_id": position.id,
                            "rationale": "want to bank early",
                        }
                    )
                    + "\n```"
                )
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                closed = store.per_trade_pnl()
                still_open = [o.id for o in store.open_positions()]

        # The fresh position is held: no close executed, position still open.
        self.assertEqual(closed, [])
        self.assertIn(position.id, still_open)

    def test_cycle_context_marks_open_position_to_live_market(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                entry = 100.0
                position = OrderRecord(
                    proposal_id="tp_seed",
                    mode="testnet",
                    symbol="BTCUSDT",
                    side=Side.BUY,
                    order_type="SPOT_LIMIT_ENTRY",
                    price=entry,
                    quantity=2.0,
                    take_profit_price=entry * 1.015,
                    stop_loss_price=entry * 0.99,
                    status=OrderStatus.POSITION_OPEN,
                )
                position.executed_qty = 2.0
                store.save_order(position)
                mark = runtime.feed.snapshot("BTCUSDT", 0).last_price
                ctx = runtime.cycle_context()

        pos = ctx["open_positions"][0]
        self.assertEqual(pos["current_price"], mark)
        self.assertEqual(pos["unrealized_pnl_usd"], round((mark - entry) * 2.0, 8))
        self.assertIsNotNone(pos["unrealized_pnl_pct"])
        self.assertIsNotNone(pos["to_take_profit_pct"])
        self.assertIsNotNone(pos["to_stop_loss_pct"])
        self.assertIsNotNone(pos["age_minutes"])

    def test_llm_sell_decision_exits_open_position(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                position = OrderRecord(
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
                store.save_order(position)
                runtime.config.risk.min_hold_hours = 0.0  # this test exercises the sell path, not the cooldown
                reply = (
                    "```json\n"
                    + json.dumps(
                        {
                            "action": "SELL",
                            "symbol": "BTCUSDT",
                            "target_order_id": position.id,
                            "rationale": "thesis done, bank it",
                        }
                    )
                    + "\n```"
                )
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                closed = store.per_trade_pnl()

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["id"], position.id)
        self.assertEqual(closed[0]["closed_by"], "SUPERVISOR_CLOSE")

    def test_llm_adjust_decision_moves_bracket(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with Store(root / "agent.sqlite3") as store:
                runtime = self._runtime(root, store, llm_settings())
                position = OrderRecord(
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
                store.save_order(position)
                reply = (
                    "```json\n"
                    + json.dumps(
                        {
                            "action": "ADJUST",
                            "symbol": "BTCUSDT",
                            "target_order_id": position.id,
                            "new_take_profit_price": 2_000_000.0,
                            "new_stop_loss_price": 0.0000005,
                            "rationale": "widen bracket",
                        }
                    )
                    + "\n```"
                )
                runtime.build_deep_agent = lambda tools=None, **_kw: FakeDeepAgent(reply)  # type: ignore[method-assign]
                runtime.run_once(cycle=1, symbols=["BTCUSDT"])
                adjusted = [o for o in store.open_positions() if o.id == position.id][0]

        self.assertEqual(adjusted.take_profit_price, 2_000_000.0)
        self.assertEqual(adjusted.stop_loss_price, 0.0000005)


if __name__ == "__main__":
    unittest.main()
