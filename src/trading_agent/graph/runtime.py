"""SupervisorRuntime: the facade that wires services into the cycle graph.

The heavy lifting lives in the sibling modules:
- nodes.py     - node implementations for the cycle pipeline
- edges.py     - graph wiring and conditional routing
- compile.py   - StateGraph assembly
- deep_agent.py- supervisor/subagent construction and prompts
- streaming.py - stream chunk handling and observability events
- state.py     - graph state types and AgentRunResult
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from trading_agent.agents import PositionReviewAgent, StrategyAgent, default_agents
from trading_agent.agents.introductions import agent_introduction_payload
from trading_agent.core.config import AppConfig, Settings, load_settings
from trading_agent.core.decision import DecisionAction, parse_supervisor_decisions
from trading_agent.core.exchange_sync import ExchangeReconciler
from trading_agent.core.exit_ladder import build_exit_plan
from trading_agent.core.logging import get_chat_logger, get_logger
from trading_agent.core.models import OrderRecord, OrderStatus, TradeIntent, TradeProposal, new_id, utc_iso
from trading_agent.core.pnl import split_symbol, unrealized_pnl
from trading_agent.core.risk import RiskGovernor
from trading_agent.core.storage import Store
from trading_agent.exchange import BinanceSpotAdapter
from trading_agent.graph import deep_agent as deep_agent_module
from trading_agent.graph.checkpointer import CheckpointerFactory
from trading_agent.graph.compile import build_cycle_graph
from trading_agent.graph.nodes import CycleNodes
from trading_agent.graph.state import AgentRunResult, EventCallback
from trading_agent.graph.streaming import astream_deep_agent, extract_agent_message
from trading_agent.prompts import PROMPTS
from trading_agent.utils.aioloop import run_coro_blocking
from trading_agent.utils.binance_skills import BinanceSkillRegistry
from trading_agent.utils.feeds import SimulatedMarketFeed
from trading_agent.utils.live_feed import BinanceLiveFeed
from trading_agent.utils.llm_factory import create_model, model_identifier, require_model_api_key
from trading_agent.utils.market_data import atr_value
from trading_agent.utils.mcp_tools import MCPToolLoader, load_mcp_config, load_mcp_tool_allowlist
from trading_agent.utils.ops_tools import build_ops_tools

__all__ = ["AgentRunResult", "SupervisorRuntime"]

LOGGER = get_logger("runtime")
CHAT_LOGGER = get_chat_logger()


def _age_minutes(opened_at: str | None) -> float | None:
    """Minutes since a position was opened, or None if the timestamp is unusable."""
    if not opened_at:
        return None
    try:
        parsed = datetime.fromisoformat(opened_at)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return round((datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds() / 60, 2)

PUBLIC_MARKET_DATA_BASE_URL = "https://api.binance.com/api"


class SupervisorRuntime:
    # Class-level: one deep-agent invocation at a time across ALL runtime
    # instances. The REPL builds separate runtimes for the cycle loop and for
    # /chat; both write the same conversation thread in checkpoints.sqlite3
    # and must never interleave.
    _invoke_lock = threading.Lock()

    def __init__(
        self,
        config: AppConfig,
        store: Store,
        *,
        settings: Settings | None = None,
        mcp_loader: MCPToolLoader | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or load_settings()
        if self.settings.enable_llm_supervisor:
            require_model_api_key(self.settings)
        self.config.model.provider = self.settings.model_provider
        self.config.model.model = self.settings.resolved_model_name()
        self.store = store
        self.binance_skills = BinanceSkillRegistry()
        if self.settings.live_data:
            self.feed: Any = BinanceLiveFeed(
                self.binance_skills,
                SimulatedMarketFeed(),
                adapter=BinanceSpotAdapter(base_url=PUBLIC_MARKET_DATA_BASE_URL, settings=self.settings),
            )
            self.signal_agents = default_agents(self.feed, self.binance_skills, enable_web_news=True)
        else:
            self.feed = SimulatedMarketFeed()
            self.signal_agents = default_agents(self.feed)
        self.strategy = StrategyAgent()
        self.position_review = PositionReviewAgent()
        self.risk = RiskGovernor()
        self.exchange_sync = ExchangeReconciler(
            store,
            self.settings,
            exit_config=config.exits,
            bid_ttl_minutes=config.risk.bid_ttl_minutes,
        )
        self.mcp_loader = mcp_loader or MCPToolLoader(
            load_mcp_config(config.home),
            tool_allowlist=load_mcp_tool_allowlist(config.home),
        )
        # Persistent agent memory; opened per top-level async call (loop-bound).
        self.checkpointer_factory = CheckpointerFactory(config.home)
        # Fallback for direct build_deep_agent() calls outside an async entry
        # point (tests, ad-hoc tooling) where no SQLite saver is open.
        self._fallback_checkpointer = InMemorySaver()
        self._active_checkpointer: Any | None = None
        self.project_root = Path(__file__).resolve().parents[3]
        self.event_callback: EventCallback | None = None

    def _emit(self, kind: str, agent: str, data: dict[str, Any]) -> None:
        """Forward an observability event to the REPL (or any subscriber)."""
        if self.event_callback is None:
            return
        try:
            self.event_callback(kind, agent, data)
        except Exception:  # observer failures must never break the cycle
            LOGGER.exception("event callback failed kind=%s agent=%s", kind, agent)

    # ------------------------------------------------------------------ #
    # cycle entry points

    def run_once(
        self,
        *,
        cycle: int = 0,
        symbols: list[str] | None = None,
        thread_id: str | None = None,
    ) -> AgentRunResult:
        """Sync shim over arun_once for the CLI and tests."""
        return self._run_sync(self.arun_once(cycle=cycle, symbols=symbols, thread_id=thread_id))

    async def arun_once(
        self,
        *,
        cycle: int = 0,
        symbols: list[str] | None = None,
        thread_id: str | None = None,
    ) -> AgentRunResult:
        active_symbols = symbols or self.config.risk.allowed_symbols
        active_thread = self._resolve_thread_id(thread_id)
        LOGGER.info(
            "run_once starting cycle=%s thread_id=%s symbols=%s",
            cycle,
            active_thread,
            ",".join(active_symbols),
        )

        nodes = CycleNodes(self)
        graph = build_cycle_graph(nodes)
        with self._invoke_lock:
            try:
                async with self.checkpointer_factory.open() as checkpointer:
                    self._active_checkpointer = checkpointer
                    try:
                        output = await graph.ainvoke(
                            {"cycle": cycle, "symbols": active_symbols, "thread_id": active_thread}
                        )
                    finally:
                        self._active_checkpointer = None
            except Exception as exc:
                LOGGER.exception("cycle failed cycle=%s", cycle)
                if nodes.active_run_id is not None:
                    self.store.finish_agent_run(nodes.active_run_id, "failed", str(exc))
                    self.store.set_heartbeat(
                        {
                            "status": "failed",
                            "thread_id": active_thread,
                            "run_id": nodes.active_run_id,
                            "cycle": cycle,
                            "error": str(exc),
                        }
                    )
                raise
        result = AgentRunResult(**output["result"])
        LOGGER.info(
            "run_once finished cycle=%s run_id=%s evidence=%s decisions=%s approved=%s rejected=%s submitted=%s errors=%s",
            result.cycle,
            result.run_id,
            result.evidence_count,
            result.decision_count,
            result.approved_count,
            result.rejected_count,
            result.submitted_trades,
            result.error_count,
        )
        return result

    def monitor_open_positions(self) -> dict[str, Any]:
        """Fast, cheap exit check between decision cycles (no LLM, no research).

        Runs the SAME deterministic exit machinery as a cycle — the deterministic
        reconcile and the exchange tiered ladder/virtual stop — but ONLY for
        symbols with an open position, so TP/SL touches are acted on within
        `bracket_monitor_seconds` instead of waiting for the next hourly cycle.
        Holds the invocation lock so it can never interleave with a cycle that is
        concurrently opening/closing positions. Never raises into the caller.
        """
        try:
            open_orders = self.store.open_positions()
        except Exception:
            LOGGER.exception("monitor: failed to load open positions")
            return {"checked": 0, "exits": 0}
        symbols = sorted({o.symbol for o in open_orders})
        if not symbols:
            return {"checked": 0, "exits": 0}
        exits = 0
        with self._invoke_lock:
            try:
                snapshots = {symbol: self.feed.snapshot(symbol, 0) for symbol in symbols}
                if self.settings.live_data:
                    atr_interval = self.config.risk.atr_interval
                    for symbol, snap in snapshots.items():
                        try:
                            snap.atr = atr_value(symbol, interval=atr_interval)
                        except Exception:  # ATR is best-effort; trailing falls back to pct
                            pass
                if self.settings.trading_agent_execution_mode in {"testnet", "live"}:
                    exits += len(self.exchange_sync.reconcile(snapshots))
            except Exception:
                LOGGER.exception("monitor: exit reconcile failed symbols=%s", ",".join(symbols))
                return {"checked": len(symbols), "exits": exits}
        if exits:
            LOGGER.info("bracket monitor acted symbols=%s exits=%s", ",".join(symbols), exits)
        return {"checked": len(symbols), "exits": exits}

    def introduce(self, *, symbols: list[str] | None = None, thread_id: str | None = None) -> dict[str, Any]:
        """Sync shim over aintroduce for the CLI and tests."""
        if not self.settings.enable_llm_supervisor:
            return agent_introduction_payload(self.config, self.settings)
        return self._run_sync(self.aintroduce(symbols=symbols, thread_id=thread_id))

    async def aintroduce(
        self, *, symbols: list[str] | None = None, thread_id: str | None = None
    ) -> dict[str, Any]:
        if not self.settings.enable_llm_supervisor:
            return agent_introduction_payload(self.config, self.settings)
        require_model_api_key(self.settings)

        active_symbols = symbols or self.config.risk.allowed_symbols
        active_thread = self._resolve_thread_id(thread_id)
        LOGGER.info("introduce starting thread_id=%s symbols=%s", active_thread, ",".join(active_symbols))

        static_context = agent_introduction_payload(self.config, self.settings)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": deep_agent_module.build_intro_prompt(self.settings, active_symbols),
                }
            ],
            "files": self._skill_state_files(),
        }
        with self._invoke_lock:
            async with self.checkpointer_factory.open() as checkpointer:
                self._active_checkpointer = checkpointer
                try:
                    agent = self.build_deep_agent([])
                    response = await agent.ainvoke(
                        payload, config={"configurable": {"thread_id": active_thread}}
                    )
                finally:
                    self._active_checkpointer = None
        message = extract_agent_message(response)
        LOGGER.info("introduce completed source=deep_agent message_chars=%s", len(message))
        self.store.save_prompt_log(
            run_id=None,
            agent_name="supervisor",
            prompt_name="deep_agent_introduce",
            prompt_version=PROMPTS["supervisor"].version,
            input_summary=f"symbols={','.join(active_symbols)}",
            output_summary=message[:1000],
        )
        return {
            "source": "deep_agent",
            "model": model_identifier(self.settings),
            "mode": self.config.mode,
            "default_symbols": list(active_symbols),
            "environment": self.settings.redacted(),
            "message": message,
            "next_commands": static_context["next_commands"],
        }

    # ------------------------------------------------------------------ #
    # operator chat

    def chat(self, message: str, *, thread_id: str | None = None) -> dict[str, Any]:
        """Sync shim over achat for the REPL."""
        return self._run_sync(self.achat(message, thread_id=thread_id))

    async def achat(self, message: str, *, thread_id: str | None = None) -> dict[str, Any]:
        """Talk to the supervisor team on the shared conversation thread.

        Returns {"message": <assistant markdown>, "decisions": [...], "errors": [...]}.
        Any BUY/SELL/CLOSE/ADJUST decision in the reply is parsed but NOT
        executed here — the caller must gate and confirm it
        (see execute_operator_decision).
        """
        if not self.settings.enable_llm_supervisor:
            raise RuntimeError(
                "chat requires TRADING_AGENT_ENABLE_LLM_SUPERVISOR=true and a model API key"
            )
        require_model_api_key(self.settings)
        active_thread = self._resolve_thread_id(thread_id)
        context = self.cycle_context()
        ops_tools = build_ops_tools(self.config.database_path, self.settings)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": deep_agent_module.build_chat_prompt(message, context),
                }
            ],
            "files": self._skill_state_files(),
        }
        invoke_config = {"configurable": {"thread_id": active_thread}}
        with self._invoke_lock:
            async with self.checkpointer_factory.open() as checkpointer:
                self._active_checkpointer = checkpointer
                try:
                    agent = self.build_deep_agent(ops_tools)
                    if self.event_callback is not None:
                        response = await astream_deep_agent(agent, payload, invoke_config, self._emit)
                    else:
                        response = await agent.ainvoke(payload, config=invoke_config)
                finally:
                    self._active_checkpointer = None
        reply = extract_agent_message(response)
        decisions, errors = parse_supervisor_decisions(reply, source="operator_chat")
        actionable = [d for d in decisions if d.action != DecisionAction.WAIT]
        self.store.save_prompt_log(
            run_id=None,
            agent_name="supervisor",
            prompt_name="deep_agent_chat",
            prompt_version=PROMPTS["supervisor"].version,
            input_summary=message[:500],
            output_summary=reply[:1000],
        )
        CHAT_LOGGER.info("operator thread=%s: %s", active_thread, message)
        CHAT_LOGGER.info("supervisor thread=%s reply:\n%s", active_thread, reply)
        for decision in actionable:
            CHAT_LOGGER.info(
                "parsed decision id=%s action=%s symbol=%s confidence=%.4f target_order_id=%s",
                decision.id,
                decision.action.value,
                decision.symbol,
                decision.confidence,
                decision.target_order_id,
            )
        for error in errors:
            CHAT_LOGGER.warning("decision parse issue: %s", error)
        return {"message": reply, "decisions": actionable, "errors": errors}

    def execute_operator_decision(self, decision: Any) -> dict[str, Any]:
        outcome = self._execute_operator_decision(decision)
        CHAT_LOGGER.info(
            "operator decision executed id=%s action=%s symbol=%s outcome=%s",
            decision.id,
            decision.action.value,
            decision.symbol,
            json.dumps(outcome, sort_keys=True, default=str),
        )
        return outcome

    def _execute_operator_decision(self, decision: Any) -> dict[str, Any]:
        """Gate and execute one operator-confirmed chat decision.

        Runs the SAME deterministic pipeline as a cycle: fresh snapshots and
        signal evidence, pre-trade blockers, RiskGovernor — then execution
        (operator_initiated relaxes only the autonomous-live-orders flag,
        never the risk gate).

        Holds _invoke_lock for the whole gate+execute sequence: the open-symbols
        and position-count reads must not race a cycle that is concurrently
        opening positions, so an operator order waits for the cycle boundary.
        """
        nodes = CycleNodes(self)
        run_id = "operator_chat"
        with self._invoke_lock:
            snapshot = self.feed.snapshot(decision.symbol, 0)
            snapshots = {decision.symbol: snapshot}
            if decision.action == DecisionAction.BUY:
                evidence = [agent.analyze(decision.symbol, snapshot, 0) for agent in self.signal_agents]
                self.store.save_evidence(evidence)
                evidence_by_id = {record.id: record for record in evidence}
                # Operator-confirmed BUYs are gated on the deterministic evidence
                # we just gathered, not the LLM's (possibly fabricated) refs, so
                # bind the decision to those real ids before the evidence-binding
                # gate runs.
                live_ids = [record.id for record in evidence if not record.is_placeholder]
                if live_ids:
                    decision.evidence_refs = live_ids
                open_symbols = {order.symbol for order in self.store.open_positions()}
                outcome = nodes._gate_buy(decision, run_id, snapshots, evidence_by_id, open_symbols)
                if outcome is None:
                    return {"executed": False, "reason": "rejected by deterministic risk gate"}
                intent, proposal = outcome
                order = self._submit_exchange_limit_entry(
                    proposal, intent, run_id, operator_initiated=True
                )
                order.decision_id = decision.id
                self.store.save_order(order)
                self.store.save_supervisor_decision(
                    decision_id=decision.id,
                    run_id=run_id,
                    action=decision.action.value,
                    symbol=decision.symbol,
                    payload=decision.summary_payload(),
                    gate_approved=True,
                    gate_reasons=["approved"],
                    executed_order_id=order.id,
                    source=decision.source,
                )
                return {"executed": True, "order_id": order.id, "mode": order.mode}
            if decision.action in {DecisionAction.CLOSE, DecisionAction.SELL}:
                nodes._execute_close(decision, snapshots, run_id)
                return {"executed": True, "order_id": decision.target_order_id}
            if decision.action == DecisionAction.ADJUST:
                nodes._execute_adjust(decision, run_id)
                return {"executed": True, "order_id": decision.target_order_id}
        return {"executed": False, "reason": f"action {decision.action.value} is not executable"}

    # ------------------------------------------------------------------ #
    # deep agent construction (delegates to graph.deep_agent)

    def subagent_specs(
        self, tools: list[Any] | None = None, *, only: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return deep_agent_module.subagent_specs(
            tools=tools,
            skills=self._skill_source_paths(),
            subagent_models=self._subagent_model_instances(),
            only=only,
        )

    def _subagent_model_instances(self) -> dict[str, Any]:
        """Build a chat-model instance per subagent override. Instances (not bare
        "provider:model" strings) are passed to deepagents so the Azure path uses
        the endpoint/api-version/key from Settings (create_model), which a string
        resolved by init_chat_model would not pick up. A failed override is logged
        and dropped so that subagent simply inherits the supervisor's model."""
        # Share ONE model instance per distinct identifier so N subagents on the
        # same deployment reuse a single httpx async client (deepagents already
        # shares the supervisor model across default subagents this way). Building
        # a separate client per subagent multiplied the "Event loop is closed"
        # cleanup tasks on Windows' per-cycle asyncio.run.
        by_identifier: dict[str, Any] = {}
        instances: dict[str, Any] = {}
        for name, identifier in self._subagent_models().items():
            if identifier not in by_identifier:
                try:
                    by_identifier[identifier] = create_model(
                        self.settings, identifier_override=identifier
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "subagent model override failed name=%s identifier=%s error=%s; "
                        "inheriting supervisor model",
                        name,
                        identifier,
                        exc,
                    )
                    continue
            instances[name] = by_identifier[identifier]
        return instances

    def _subagent_models(self) -> dict[str, str]:
        """Per-subagent model overrides: config.json `model.subagent_models`,
        then the TRADING_AGENT_SUBAGENT_MODELS env JSON on top."""
        merged: dict[str, str] = dict(self.config.model.subagent_models or {})
        if self.settings.subagent_models_json:
            try:
                merged.update(json.loads(self.settings.subagent_models_json))
            except (TypeError, ValueError) as exc:
                LOGGER.warning("ignoring invalid TRADING_AGENT_SUBAGENT_MODELS: %s", exc)
        unknown = sorted(set(merged) - set(deep_agent_module.SUBAGENT_NAMES))
        for name in unknown:
            LOGGER.warning("ignoring subagent model override for unknown agent %s", name)
            merged.pop(name)
        return merged

    def build_deep_agent(self, tools: list[Any] | None = None, *, tier: str = "FULL") -> Any:
        # extra_tools reaches the ROOT supervisor, not just the subagents:
        # /chat's ops tools (list_open_orders etc.) must be directly callable
        # by the supervisor — its prompt instructs it to use them.
        # REVIEW tier: cheaper model + only the review subagents (manage-only).
        if tier == "REVIEW" and self.config.cost.quiet_model:
            model = create_model(self.settings, identifier_override=self.config.cost.quiet_model)
            subagents = self.subagent_specs(tools, only=self.config.cost.review_subagents)
        else:
            model = self._deep_agent_model()
            subagents = self.subagent_specs(tools)
        return deep_agent_module.build_deep_agent(
            model=model,
            subagents=subagents,
            skills=self._skill_source_paths(),
            checkpointer=self._active_checkpointer or self._fallback_checkpointer,
            extra_tools=tools,
        )

    # ------------------------------------------------------------------ #
    # async plumbing

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Persistent per-thread loop (not asyncio.run, which closes the loop
            # each call and orphans httpx async-client cleanup -> "Event loop is
            # closed" spam on Windows). See utils.aioloop.
            return run_coro_blocking(coro)
        raise RuntimeError(
            "called a sync runtime entry point from a running event loop; "
            "use the async variant (arun_once/aintroduce/achat) instead"
        )

    def _resolve_thread_id(self, thread_id: str | None) -> str:
        """Resolve and persist the conversation thread id.

        Rotates the stored thread id once when upgrading from the in-memory
        checkpointer era: old ids resolve to empty SQLite checkpoints, which is
        harmless but confusing in logs.
        """
        if thread_id is None:
            stored = self.store.get_setting("agent_thread_id", None)
            backend = self.store.get_setting("checkpointer_backend", None)
            if stored and backend != "sqlite":
                LOGGER.info("rotating agent thread id for sqlite checkpointer (was %s)", stored)
                stored = None
            thread_id = stored or new_id("thread")
        self.store.set_setting("agent_thread_id", thread_id)
        self.store.set_setting("checkpointer_backend", "sqlite")
        return thread_id

    def _deep_agent_model(self) -> Any:
        return create_model(self.settings)

    def _skill_source_paths(self) -> list[str]:
        return deep_agent_module.skill_source_paths(self.project_root, self.binance_skills)

    def _skill_state_files(self) -> dict[str, dict[str, str]]:
        return deep_agent_module.skill_state_files(self.project_root, self._skill_source_paths())

    # ------------------------------------------------------------------ #
    # account and context services

    def account_balances(self) -> list[dict[str, Any]] | None:
        """Best-effort exchange balance fetch; None when unavailable."""
        if self.settings.binance_api_key is None or self.settings.binance_api_secret is None:
            return None
        try:
            credentials = BinanceSpotAdapter.credentials_from_env(settings=self.settings)
            adapter = BinanceSpotAdapter(base_url=self.settings.exchange_base_url(), settings=self.settings)
            return adapter.account_balances(credentials)
        except Exception as exc:
            LOGGER.warning("balance fetch failed: %s", exc)
            return None

    def available_quote_balance(self, symbol: str) -> float | None:
        """Available quote-asset balance from the configured execution environment."""
        balances = self.account_balances()
        if balances is None:
            return None
        _, quote_asset = split_symbol(symbol)
        for item in balances:
            if item["asset"].upper() == quote_asset:
                return float(item.get("free", 0.0) or 0.0)
        return 0.0

    def balance_snapshot(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Fresh Binance account balances for audit logs after exchange actions."""
        balances = self.account_balances()
        if balances is None:
            return None
        payload: dict[str, Any] = {
            "venue": self.settings.binance_venue,
            "base_url": self.settings.exchange_base_url(),
            "balances": balances,
        }
        if symbol:
            base_asset, quote_asset = split_symbol(symbol)
            by_asset = {item["asset"].upper(): item for item in balances}
            payload["symbol"] = symbol
            payload["base_asset"] = base_asset
            payload["quote_asset"] = quote_asset
            payload["base_balance"] = by_asset.get(base_asset, {"asset": base_asset, "free": 0.0, "locked": 0.0})
            payload["quote_balance"] = by_asset.get(quote_asset, {"asset": quote_asset, "free": 0.0, "locked": 0.0})
        return payload

    @staticmethod
    def _exit_ladder_view(order: Any) -> dict[str, Any] | None:
        """Compact tiered-exit state for the supervisor: which TP tiers have
        banked, whether the runner is riding, and the current (ratcheting) stop.
        The deterministic ladder manages scale-out, so the LLM should not
        full-CLOSE a winner that is still riding — only on a thesis break."""
        plan = getattr(order, "exit_plan", None)
        if plan is None:
            return None
        return {
            "tiered": plan.tiered,
            "tiers_filled": plan.tiers_filled,
            "total_tiers": len(plan.legs),
            "runner_active": plan.runner_active,
            "runner_size_pct": plan.runner_size_pct,
            "current_stop_price": plan.current_stop_price,
            "high_water_price": plan.high_water_price,
            "legs": [
                {
                    "tier": leg.tier,
                    "target_price": leg.target_price,
                    "size_pct": leg.size_pct,
                    "filled": leg.filled,
                }
                for leg in plan.legs
            ],
        }

    def _open_position_view(self, order: Any) -> dict[str, Any]:
        """One open position marked to the LIVE market so the supervisor can
        decide HOLD vs CLOSE/SELL: current price, unrealized PnL (absolute and
        %), distance to TP/SL, and age. Mark fields are None if the feed is
        unavailable (the decision context degrades, it does not crash)."""
        entry = order.avg_fill_price or order.price
        view: dict[str, Any] = {
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "entry_price": order.price,
            "avg_fill_price": order.avg_fill_price,
            "quantity": order.quantity,
            "executed_qty": order.executed_qty,
            "exchange_status": order.exchange_status,
            "take_profit_price": order.take_profit_price,
            "stop_loss_price": order.stop_loss_price,
            "status": order.status.value,
            # Tiered scale-out state so the supervisor reasons about the live
            # ladder instead of assuming a single all-or-nothing take-profit.
            "exit_ladder": self._exit_ladder_view(order),
            "opened_at": order.opened_at,
            "current_price": None,
            "unrealized_pnl_usd": None,
            "unrealized_pnl_pct": None,
            "to_take_profit_pct": None,
            "to_stop_loss_pct": None,
            "age_minutes": _age_minutes(order.opened_at),
        }
        try:
            mark = self.feed.snapshot(order.symbol, 0).last_price
        except Exception as exc:  # feed hiccup must not break the cycle context
            LOGGER.warning("mark-to-market failed for %s: %s", order.symbol, exc)
            return view
        view["current_price"] = mark
        if order.status == OrderStatus.POSITION_OPEN and entry:
            # Marks only the HELD portion (subtracts filled TP tiers), so a
            # scaled-out position is not shown as exposed on its full original size.
            unrealized_usd, unrealized_pct = unrealized_pnl(order, mark)
            view["unrealized_pnl_usd"] = unrealized_usd
            view["unrealized_pnl_pct"] = unrealized_pct
        if mark:
            if order.take_profit_price:
                view["to_take_profit_pct"] = round((order.take_profit_price / mark - 1) * 100, 4)
            if order.stop_loss_price:
                view["to_stop_loss_pct"] = round((mark / order.stop_loss_price - 1) * 100, 4)
        return view

    def cycle_context(self) -> dict[str, Any]:
        """State the agents must consider before deciding anything new.

        Includes account balances, currently open positions (with order ids the
        supervisor can target with CLOSE/ADJUST), recent per-trade realized PnL,
        and the previous cycle's checkpoint - so every decision starts from
        what already happened, not a blank slate.
        """
        open_positions = [self._open_position_view(order) for order in self.store.open_positions()]
        # The exchange (esp. the testnet faucet) can report hundreds of dust
        # balances; dumping all of them costs ~10k tokens/cycle of pure noise. The
        # supervisor only needs cash (USDT) + the tradable universe + whatever it
        # actually holds, so filter to those assets and note how many were hidden.
        all_balances = self.account_balances()
        balances = all_balances
        balances_note = None
        if all_balances:
            relevant: set[str] = {"USDT"}
            for symbol in list(self.config.risk.allowed_symbols) + [
                str(view.get("symbol", "")) for view in open_positions
            ]:
                if symbol:
                    base, quote = split_symbol(symbol)
                    relevant.update((base, quote))
            balances = sorted(
                (b for b in all_balances if str(b.get("asset")) in relevant),
                key=lambda b: str(b.get("asset")),
            )
            hidden = len(all_balances) - len(balances)
            if hidden > 0:
                balances_note = (
                    f"{len(balances)} tradable-universe/held assets shown; "
                    f"{hidden} other venue balances (testnet faucet dust) hidden"
                )
        per_trade = self.store.per_trade_pnl(10)
        summary = self.store.summary()
        now = datetime.now(UTC)
        return {
            "now": {
                "utc": now.isoformat(timespec="seconds"),
                "date": now.strftime("%Y-%m-%d"),
                "weekday": now.strftime("%A"),
                "year": now.year,
                "month": now.strftime("%B"),
                "time_utc": now.strftime("%H:%M"),
            },
            "execution_mode": self.settings.trading_agent_execution_mode,
            "account_balances": balances,
            "account_balances_note": balances_note,
            "open_positions": open_positions,
            "open_position_count": len(open_positions),
            "max_open_positions": self.config.risk.max_open_positions,
            "recent_closed_trades": per_trade,
            # Memory loop: what recent trades actually did + the desk's realized
            # win-rate / average-R / expectancy, so conviction is grounded in
            # outcomes rather than starting from a blank slate each cycle.
            "recent_reflections": self.store.recent_reflections(8),
            "trade_stats": self.store.trade_stats(),
            "realized_pnl_total": summary.get("realized_pnl"),
            # Breaches = genuine risk-control events (alarming). Vetoes = the gate
            # healthily declining marginal trades (expected; NOT a reason to stand
            # down). Surfaced separately so reporting stops reading routine vetoes
            # as breaches (the chat.log "3 breaches, stand down" failure mode).
            "risk_breaches_total": summary.get("risk_breaches"),
            "gate_vetoes_total": summary.get("vetoes"),
            "previous_cycle": self.store.get_setting("agent_checkpoint", None),
            "kill_switch": bool(self.store.get_setting("kill_switch", False)),
            # Static higher-timeframe brief built once per UTC day (1M..15m). The
            # past is fixed: study it on day-open, then trade off current data.
            "daily_brief": self.store.get_setting("daily_brief", None),
        }

    # ------------------------------------------------------------------ #
    # order submission (exchange-facing)

    def _check_exchange_execution_allowed(self, *, operator_initiated: bool = False) -> str:
        """Validate the multi-flag opt-in for the active execution mode; returns the mode."""
        mode = self.settings.trading_agent_execution_mode
        if mode == "testnet":
            if not self.settings.enable_testnet_orders:
                raise RuntimeError("TRADING_AGENT_ENABLE_TESTNET_ORDERS=true is required for testnet execution")
            if self.settings.binance_venue != "testnet":
                raise RuntimeError("testnet execution requires BINANCE_VENUE=testnet")
            if "testnet.binance.vision" not in self.settings.exchange_base_url():
                raise RuntimeError("testnet execution requires the testnet.binance.vision base URL")
            return mode
        if mode == "live":
            blockers = self.settings.live_order_blockers()
            if not self.config.live.enabled:
                blockers.append("config.json live.enabled must be true")
            if not self.config.live.venue_confirmed:
                blockers.append("config.json live.venue_confirmed must be true")
            if not operator_initiated and not self.config.live.auto_orders_within_caps:
                blockers.append(
                    "autonomous live orders are disabled (live.auto_orders_within_caps=false); "
                    "use /chat to place operator-confirmed orders"
                )
            if blockers:
                raise RuntimeError("live execution blocked: " + "; ".join(blockers))
            return mode
        raise RuntimeError(f"execution mode {mode} does not submit exchange orders")

    def _submit_exchange_limit_entry(
        self,
        proposal: TradeProposal,
        intent: TradeIntent,
        run_id: str,
        *,
        operator_initiated: bool = False,
    ) -> OrderRecord:
        mode = self._check_exchange_execution_allowed(operator_initiated=operator_initiated)
        credentials = BinanceSpotAdapter.credentials_from_env(settings=self.settings)
        adapter = BinanceSpotAdapter(base_url=self.settings.exchange_base_url(), settings=self.settings)
        client_order_id = f"ta{intent.id.replace('_', '')[:24]}"
        # Round to the symbol's tickSize/stepSize: unquantized values are
        # rejected by Binance with -1013 PRICE_FILTER / LOT_SIZE failures.
        quantity, price = adapter.quantize_order(proposal.symbol, proposal.quantity, proposal.price)
        executed_price = float(price)
        executed_quantity = float(quantity)
        # Tiered scale-out ladder (TP1/TP2/runner + ratcheting stop) sized off
        # the entry; legacy single-leg bracket when exits.enabled is False.
        exit_plan = build_exit_plan(
            executed_price,
            self.config.exits,
            fallback_take_profit_pct=proposal.take_profit_pct,
            fallback_stop_loss_pct=proposal.stop_loss_pct,
            # Demand-zone bids carry an absolute stop just below the zone; honor it
            # instead of the fixed config stop so the invalidation matches the chart.
            stop_price=proposal.stop_price,
        )
        # Persist the row BEFORE the exchange call: if the process dies between
        # submit and save, the PENDING_SUBMIT row's client_order_id lets the
        # reconciler adopt or discard the order on restart instead of orphaning
        # a live position.
        order = OrderRecord(
            proposal_id=proposal.id,
            mode=mode,
            symbol=proposal.symbol,
            side=proposal.side,
            order_type=f"SPOT_{mode.upper()}_LIMIT_ENTRY",
            price=executed_price,
            quantity=executed_quantity,
            take_profit_price=exit_plan.legs[0].target_price,
            stop_loss_price=exit_plan.current_stop_price,
            exit_plan=exit_plan,
            status=OrderStatus.PENDING_SUBMIT,
            client_order_id=client_order_id,
        )
        self.store.save_order(order)
        LOGGER.info(
            "submitting %s spot LIMIT entry run_id=%s intent_id=%s symbol=%s side=%s quantity=%s price=%s client_order_id=%s",
            mode,
            run_id,
            intent.id,
            proposal.symbol,
            proposal.side.value,
            quantity,
            price,
            client_order_id,
        )
        response = adapter.submit_limit_order(
            credentials,
            proposal.symbol,
            proposal.side.value,
            quantity,
            price,
            client_order_id=client_order_id,
        )
        raw = response.get("raw", {})
        balances_after_submit = self.balance_snapshot(proposal.symbol)
        if raw.get("orderId") is None:
            # Leave the row PENDING_SUBMIT: the reconciler will query the venue
            # by client_order_id and adopt or discard it.
            self.store.log_event(
                "exchange_entry_response_malformed",
                {
                    "order_id": order.id,
                    "client_order_id": client_order_id,
                    "response": raw,
                    "balances_after_submit": balances_after_submit,
                },
            )
            raise RuntimeError(
                f"exchange response for {proposal.symbol} lacks orderId; "
                f"order left PENDING_SUBMIT as {order.id} for reconciliation"
            )
        order.status = OrderStatus.ENTRY_OPEN
        order.exchange_order_id = str(raw["orderId"])
        order.exchange_status = raw.get("status")
        order.executed_qty = float(raw.get("executedQty", 0) or 0)
        order.cumulative_quote_qty = float(raw.get("cummulativeQuoteQty", 0) or 0)
        order.last_synced_at = utc_iso()
        self.store.save_order(order)
        LOGGER.info(
            "%s spot order submitted run_id=%s order_id=%s exchange_order_id=%s symbol=%s status=%s",
            mode,
            run_id,
            order.id,
            raw.get("orderId"),
            proposal.symbol,
            raw.get("status"),
        )
        if balances_after_submit is not None:
            quote = balances_after_submit.get("quote_balance", {})
            base = balances_after_submit.get("base_balance", {})
            LOGGER.info(
                "balances after %s entry submit symbol=%s %s_free=%s %s_locked=%s %s_free=%s %s_locked=%s",
                mode,
                proposal.symbol,
                quote.get("asset"),
                quote.get("free"),
                quote.get("asset"),
                quote.get("locked"),
                base.get("asset"),
                base.get("free"),
                base.get("asset"),
                base.get("locked"),
            )
        self.store.log_event(
            f"{mode}_spot_entry_submitted",
            {
                "run_id": run_id,
                "order_id": order.id,
                "intent_id": intent.id,
                "proposal_id": proposal.id,
                "symbol": proposal.symbol,
                "side": proposal.side.value,
                "price": proposal.price,
                "quantity": proposal.quantity,
                "client_order_id": client_order_id,
                "exchange_response": raw,
                "balances_after_submit": balances_after_submit,
            },
        )
        return order

    # Backwards-compatible alias (testnet was the only exchange mode before live).
    _submit_testnet_limit_entry = _submit_exchange_limit_entry
