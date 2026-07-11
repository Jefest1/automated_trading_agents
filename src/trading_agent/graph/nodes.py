"""Nodes of the supervised trading cycle graph.

Topology (see edges.py):
    START -> prepare_context -> consult_agents -> parse_decision
          -> [risk_gate -> execute]? -> report -> END

The supervisor deep agent (consult_agents) is the trader: it consults every
specialist subagent and emits the final BUY/SELL/WAIT/CLOSE/ADJUST decision.
The deterministic risk gate (pre-trade blockers + RiskGovernor) can veto any
decision before execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import UsageMetadataCallbackHandler

from trading_agent.core.decision import (
    DecisionAction,
    SupervisorDecision,
    decisions_from_intents,
    parse_supervisor_decisions,
    wait_decision,
)
from trading_agent.core.logging import get_logger
from trading_agent.core.models import EvidenceRecord, OrderRecord, OrderStatus, TradeIntent, utc_iso
from trading_agent.core.pnl import split_symbol
from trading_agent.core.risk import conviction_size, maker_pullback_price
from trading_agent.graph import deep_agent as deep_agent_module
from trading_agent.graph.cadence import classify_cycle
from trading_agent.graph.state import AgentRunResult, RuntimeGraphState
from trading_agent.graph.streaming import ToolCallLogger, astream_deep_agent, extract_agent_message
from trading_agent.prompts import PROMPTS, all_prompts
from trading_agent.utils.market_data import atr_value, level_map_for, multi_timeframe_brief
from trading_agent.utils.mcp_tools import MCPToolLoadResult
from trading_agent.utils.token_cost import summarize_usage

if TYPE_CHECKING:
    from trading_agent.graph.runtime import SupervisorRuntime

LOGGER = get_logger("nodes")

# Exception type names (checked along the __cause__/__context__ chain) that mean
# the network dropped, not that the request was invalid; safe to retry.
_TRANSIENT_NETWORK_ERRORS = frozenset(
    {
        "ReadError",
        "ReadTimeout",
        "ConnectError",
        "ConnectTimeout",
        "RemoteProtocolError",
        "APIConnectionError",
        "APITimeoutError",
    }
)


def _is_transient_network_error(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in _TRANSIENT_NETWORK_ERRORS:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _order_age_minutes(opened_at: str | None) -> float | None:
    """Minutes since an order opened, or None if the timestamp is unusable."""
    if not opened_at:
        return None
    try:
        parsed = datetime.fromisoformat(opened_at)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds() / 60.0


class CycleNodes:
    """Node implementations bound to a SupervisorRuntime's services.

    Per-run working objects that must not enter graph state (live tool
    instances, the active run id for failure handling) are kept on the
    instance; cycles are serialized by the runtime's invocation lock.
    """

    def __init__(self, runtime: SupervisorRuntime) -> None:
        self.runtime = runtime
        self.active_run_id: str | None = None
        self._mcp_tools: list[Any] = []

    # ------------------------------------------------------------------ #
    # prepare_context

    def prepare_context(self, state: RuntimeGraphState) -> RuntimeGraphState:
        runtime = self.runtime
        cycle = int(state["cycle"])
        symbols = list(state["symbols"])
        thread_id = str(state["thread_id"])
        run_id = runtime.store.start_agent_run(thread_id, cycle)
        self.active_run_id = run_id
        errors: list[str] = []
        LOGGER.info("cycle started run_id=%s cycle=%s symbols=%s", run_id, cycle, ",".join(symbols))
        runtime.store.set_heartbeat(
            {"status": "running", "thread_id": thread_id, "run_id": run_id, "cycle": cycle}
        )
        self._log_prompt_registry(run_id, cycle, symbols)

        tool_result = self._load_mcp_tools(run_id)
        self._mcp_tools = tool_result.tools
        errors.extend(tool_result.errors)
        for error in tool_result.errors:
            LOGGER.warning("mcp tool load issue run_id=%s error=%s", run_id, error)

        LOGGER.info("building market snapshots run_id=%s symbols=%s", run_id, ",".join(symbols))
        snapshots = {symbol: runtime.feed.snapshot(symbol, cycle) for symbol in symbols}
        level_maps: dict[str, Any] = {}
        # ATR (on the swing atr_interval, 1h) sizes maker-pullback entries
        # (limit = bid - k*ATR) and the runner trail. Only fetch live (production
        # klines); offline/sim runs leave atr None -> min offset.
        if runtime.settings.live_data:
            atr_interval = runtime.config.risk.atr_interval
            for symbol, snapshot in snapshots.items():
                snapshot.atr = atr_value(symbol, interval=atr_interval)
            # Demand-zone map per symbol (support/resistance zones + regime) from
            # monthly..1h candles. Drives laddered support bids instead of WAIT.
            for symbol, snapshot in snapshots.items():
                lmap = level_map_for(symbol, current_price=snapshot.last_price or None)
                if lmap is not None:
                    level_maps[symbol] = lmap
                    LOGGER.info(
                        "level map run_id=%s symbol=%s regime=%s supports=%s resistances=%s",
                        run_id,
                        symbol,
                        lmap.regime,
                        len(lmap.support_zones),
                        len(lmap.resistance_zones),
                    )
            # Daily warm-up: build the compact multi-timeframe brief once per UTC
            # day (the static higher-TF context). Later same-day cycles reuse it
            # and only fetch current data. Persisted so it survives restarts.
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            if runtime.store.get_setting("last_daily_brief_date") != today:
                brief = multi_timeframe_brief(symbols)
                runtime.store.set_setting("daily_brief", {"date": today, "symbols": brief})
                runtime.store.set_setting("last_daily_brief_date", today)
                runtime.store.log_event("daily_brief_built", {"date": today, "symbols": symbols})
                LOGGER.info(
                    "daily multi-timeframe brief built run_id=%s date=%s symbols=%s",
                    run_id,
                    today,
                    ",".join(symbols),
                )
        for snapshot in snapshots.values():
            LOGGER.info(
                "market snapshot run_id=%s symbol=%s last=%s bid=%s ask=%s volume_24h=%s observed_at=%s",
                run_id,
                snapshot.symbol,
                snapshot.last_price,
                snapshot.bid_price,
                snapshot.ask_price,
                snapshot.volume_24h,
                snapshot.observed_at,
            )
            runtime._emit(
                "snapshot",
                "market_feed",
                {
                    "symbol": snapshot.symbol,
                    "last_price": snapshot.last_price,
                    "source": getattr(runtime.feed, "last_source", {}).get(snapshot.symbol, "simulated"),
                },
            )

        reconciled_count = 0
        if runtime.settings.trading_agent_execution_mode in {"testnet", "live"}:
            exchange_updated = runtime.exchange_sync.reconcile(snapshots)
            if exchange_updated:
                reconciled_count += len(exchange_updated)
                LOGGER.info("exchange orders synced run_id=%s count=%s", run_id, len(exchange_updated))
                for order in exchange_updated:
                    runtime._emit(
                        "order",
                        "exchange_sync",
                        {
                            "mode": order.mode,
                            "symbol": order.symbol,
                            "price": order.price,
                            "quantity": order.quantity,
                            "sync": True,
                            "status": order.status.value,
                            "exchange_status": order.exchange_status,
                            "executed_qty": order.executed_qty,
                            "quote_spent": order.cumulative_quote_qty,
                        },
                    )

        evidence: list[EvidenceRecord] = []
        for symbol, snapshot in snapshots.items():
            for agent in runtime.signal_agents:
                evidence.append(agent.analyze(symbol, snapshot, cycle))
        runtime.store.save_evidence(evidence)
        LOGGER.info("evidence saved run_id=%s count=%s", run_id, len(evidence))
        for record in evidence:
            runtime._emit(
                "evidence",
                record.agent,
                {
                    "symbol": record.symbol,
                    "kind": record.kind,
                    "score": record.score,
                    "confidence": record.confidence,
                    "source": record.source,
                },
            )
            LOGGER.info(
                "agent evidence run_id=%s agent=%s symbol=%s kind=%s score=%.6f confidence=%.6f source=%s",
                run_id,
                record.agent,
                record.symbol,
                record.kind,
                record.score,
                record.confidence,
                record.source,
            )

        proposals = runtime.strategy.propose(snapshots, evidence, runtime.config, level_maps)
        baseline_intents = [
            TradeIntent.from_proposal(proposal, source_agent="strategy") for proposal in proposals
        ]
        LOGGER.info(
            "strategy produced baseline run_id=%s proposals=%s", run_id, len(baseline_intents)
        )
        proposal_symbols = {intent.symbol for intent in baseline_intents}
        for intent in baseline_intents:
            runtime._emit(
                "proposal",
                "strategy",
                {
                    "symbol": intent.symbol,
                    "side": intent.side.value,
                    "limit_price": intent.limit_price,
                    "quantity": intent.quantity,
                    "confidence": intent.confidence,
                    "expected_edge_bps": intent.expected_edge_bps,
                },
            )
            runtime.store.log_event(
                "strategy_trade_intent",
                {
                    "run_id": run_id,
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "side": intent.side.value,
                    "limit_price": intent.limit_price,
                    "quantity": intent.quantity,
                    "confidence": intent.confidence,
                    "expected_edge_bps": intent.expected_edge_bps,
                },
            )
        for symbol in symbols:
            if symbol in proposal_symbols:
                continue
            wait = self._strategy_wait_diagnostics(
                symbol, [record for record in evidence if record.symbol == symbol]
            )
            LOGGER.info(
                "strategy wait run_id=%s symbol=%s reason=%s expected_edge_bps=%s min_expected_edge_bps=%s avg_confidence=%s",
                run_id,
                symbol,
                wait["reason"],
                wait.get("expected_edge_bps"),
                wait.get("min_expected_edge_bps"),
                wait.get("avg_confidence"),
            )
            runtime.store.log_event("strategy_wait", {"run_id": run_id, "symbol": symbol, **wait})

        # Manage open risk FIRST: deterministic HOLD/ADJUST/CLOSE (positions) and
        # KEEP/CANCEL (resting bids) proposals the team reviews before any new entry.
        open_orders = runtime.store.open_positions()
        position_reviews = runtime.position_review.review(
            open_orders, snapshots, level_maps, runtime.config
        )
        for review in position_reviews:
            runtime.store.log_event(
                "position_review",
                {
                    "run_id": run_id,
                    "order_id": review.order_id,
                    "symbol": review.symbol,
                    "status": review.status,
                    "recommended_action": review.recommended_action,
                    "age_minutes": review.age_minutes,
                    "min_hold_satisfied": review.min_hold_satisfied,
                    "regime": review.regime,
                    "reason": review.reason,
                },
            )
            LOGGER.info(
                "position review run_id=%s order_id=%s symbol=%s status=%s action=%s age_min=%s reason=%s",
                run_id,
                review.order_id,
                review.symbol,
                review.status,
                review.recommended_action,
                review.age_minutes,
                review.reason,
            )

        cycle_tier = self._classify_cycle(run_id, snapshots, baseline_intents, open_orders)

        return {
            "run_id": run_id,
            "mcp_tool_count": len(tool_result.tools),
            "errors": errors,
            "snapshots": snapshots,
            "evidence": evidence,
            "baseline_intents": baseline_intents,
            "level_maps": level_maps,
            "position_reviews": position_reviews,
            "reconciled_count": reconciled_count,
            "cycle_tier": cycle_tier,
        }

    def _classify_cycle(
        self,
        run_id: str,
        snapshots: dict[str, Any],
        baseline_intents: list[Any],
        open_orders: list[Any] | None = None,
    ) -> str:
        """Material-change cost tier for this cycle. Deterministic mode or
        disabled cost tiers always run FULL (there is no LLM spend to save)."""
        runtime = self.runtime
        if not runtime.settings.enable_llm_supervisor or not runtime.config.cost.enabled:
            return "FULL"
        now = datetime.now(UTC)
        today = now.strftime("%Y-%m-%d")
        is_first_of_day = runtime.store.get_setting("cadence_day", None) != today
        last_marks = runtime.store.get_setting("last_supervised_marks", {}) or {}
        last_ts = runtime.store.get_setting("last_supervised_ts", None)
        minutes_since_review: float | None = None
        if last_ts:
            try:
                parsed = datetime.fromisoformat(last_ts)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                minutes_since_review = (now - parsed).total_seconds() / 60
            except (TypeError, ValueError):
                minutes_since_review = None
        orders = open_orders if open_orders is not None else runtime.store.open_positions()
        open_views = [runtime._open_position_view(o) for o in orders]
        # Capacity: a new-entry signal only justifies FULL when a slot is available
        # (position cap + correlated-notional headroom). At capacity the entry
        # would be gate-rejected, so the cycle falls through to REVIEW.
        risk = runtime.config.risk
        correlated_open = sum(
            (o.cumulative_quote_qty or (o.price * o.quantity)) for o in orders
        )
        slot_available = len(orders) < risk.max_open_positions
        headroom_ok = (
            risk.max_correlated_notional_usd <= 0
            or (risk.max_correlated_notional_usd - correlated_open) >= runtime.config.sizing.min_notional_usd
        )
        has_entry_capacity = slot_available and headroom_ok
        tier, reason = classify_cycle(
            baseline_intents=baseline_intents,
            open_views=open_views,
            snapshots=snapshots,
            last_marks=last_marks,
            minutes_since_review=minutes_since_review,
            is_first_cycle_of_day=is_first_of_day,
            cost=runtime.config.cost,
            has_entry_capacity=has_entry_capacity,
        )
        LOGGER.info("cycle tier run_id=%s tier=%s reason=%s", run_id, tier, reason)
        runtime._emit("info", "cadence", {"tier": tier, "reason": reason})
        if tier != "SKIP":
            runtime.store.set_setting(
                "last_supervised_marks", {s: snap.last_price for s, snap in snapshots.items()}
            )
            runtime.store.set_setting("last_supervised_ts", now.isoformat())
            if tier == "FULL":
                runtime.store.set_setting("cadence_day", today)
        return tier

    # ------------------------------------------------------------------ #
    # consult_agents

    async def consult_agents(self, state: RuntimeGraphState) -> RuntimeGraphState:
        runtime = self.runtime
        if not runtime.settings.enable_llm_supervisor:
            return {"supervisor_message": ""}
        run_id = str(state["run_id"])
        cycle = int(state["cycle"])
        symbols = list(state["symbols"])
        errors = list(state.get("errors", []))
        try:
            LOGGER.info(
                "invoking deep agent supervisor run_id=%s cycle=%s symbols=%s",
                run_id,
                cycle,
                ",".join(symbols),
            )
            runtime._emit(
                "supervisor",
                "supervisor",
                {"message": f"starting research cycle {cycle}", "symbols": symbols},
            )
            tier = str(state.get("cycle_tier", "FULL"))
            context = runtime.cycle_context()
            context["quant_baseline"] = [asdict(intent) for intent in state.get("baseline_intents", [])]
            # Per-order recommendations (HOLD/CLOSE for positions, KEEP/CANCEL for
            # resting bids) reviewed before any new entry; present in both tiers.
            context["position_reviews"] = [
                review.to_dict() for review in state.get("position_reviews", [])
            ]
            if tier == "REVIEW":
                # Trim the context: a REVIEW manages open positions only, no new
                # entries, so the daily brief and evidence catalog are dropped to
                # keep the cheap cycle cheap.
                context.pop("daily_brief", None)
            else:
                # Real evidence ids gathered this cycle, so a BUY's evidence_refs can
                # cite records that actually exist (the gate rejects fabricated refs).
                context["evidence_available"] = [
                    {
                        "id": record.id,
                        "agent": record.agent,
                        "symbol": record.symbol,
                        "kind": record.kind,
                        "score": record.score,
                        "confidence": record.confidence,
                        "source": record.source,
                        "observed_at": record.observed_at,
                    }
                    for record in state.get("evidence", [])
                    if not record.is_placeholder
                ]
                # Computed demand/supply zones + regime for the technical_analyst to
                # confirm and for the supervisor to anchor zone-based bids against.
                context["level_maps"] = {
                    symbol: lmap.to_dict() for symbol, lmap in state.get("level_maps", {}).items()
                }
            if context.get("account_balances"):
                runtime._emit(
                    "info",
                    "account",
                    {
                        "message": "balances: "
                        + ", ".join(
                            f"{item['asset']}={item['free']:g}" for item in context["account_balances"][:8]
                        )
                    },
                )
            # REVIEW runs a reduced, cheaper agent with no MCP tools; FULL gets the
            # full model, all subagents, and the MCP tools.
            cycle_mcp_tools = [] if tier == "REVIEW" else self._mcp_tools
            agent = runtime.build_deep_agent(cycle_mcp_tools, tier=tier)
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": deep_agent_module.build_cycle_prompt(cycle, symbols, context, tier=tier),
                    }
                ],
                "files": runtime._skill_state_files(),
            }
            # Usage callback propagates into every nested subagent/model call
            # (subgraphs=True), so this captures the whole cycle's token spend.
            # ToolCallLogger logs each tool call by name (built-in vs MCP) so MCP
            # usage is visible in the log regardless of streaming attribution.
            usage_cb = UsageMetadataCallbackHandler()
            mcp_tool_names = {getattr(t, "name", "") for t in (cycle_mcp_tools or [])}
            invoke_config = {
                "configurable": {"thread_id": runtime.store.get_setting("agent_thread_id")},
                "callbacks": [usage_cb, ToolCallLogger(mcp_tool_names)],
            }
            # A dropped connection mid-stream (httpx ReadError etc.) cannot be
            # retried inside the SDK once the stream has started; retry the whole
            # supervisor invocation for transient network failures.
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    if runtime.event_callback is not None:
                        response = await astream_deep_agent(agent, payload, invoke_config, runtime._emit)
                    else:
                        response = await agent.ainvoke(payload, config=invoke_config)
                    break
                except Exception as exc:
                    if attempt == attempts or not _is_transient_network_error(exc):
                        raise
                    LOGGER.warning(
                        "supervisor invocation hit a transient network error "
                        "(attempt %s/%s): %s; retrying",
                        attempt,
                        attempts,
                        exc,
                    )
                    await asyncio.sleep(5 * attempt)
            self._record_token_usage(run_id, cycle, usage_cb.usage_metadata)
            message = extract_agent_message(response)
            runtime.store.save_prompt_log(
                run_id=run_id,
                agent_name="supervisor",
                prompt_name="deep_agent_invoke",
                prompt_version=PROMPTS["supervisor"].version,
                input_summary=f"cycle={cycle}; symbols={','.join(symbols)}",
                output_summary=message[:1000],
            )
            LOGGER.info(
                "deep agent supervisor completed run_id=%s response_chars=%s", run_id, len(message)
            )
            return {"supervisor_message": message, "errors": errors}
        except Exception as exc:
            LOGGER.exception("deep agent supervisor failed run_id=%s cycle=%s", run_id, cycle)
            runtime.store.save_tool_call_log(
                run_id=run_id,
                agent_name="supervisor",
                tool_name="deep_agent_invoke",
                input_summary=f"cycle={cycle}; symbols={','.join(symbols)}",
                output_summary="recoverable supervisor invocation failure",
                error=str(exc),
            )
            errors.append(str(exc))
            return {"supervisor_message": "", "errors": errors}

    # ------------------------------------------------------------------ #
    # parse_decision

    def parse_decision(self, state: RuntimeGraphState) -> RuntimeGraphState:
        runtime = self.runtime
        run_id = str(state["run_id"])
        symbols = list(state["symbols"])
        errors = list(state.get("errors", []))
        baseline = list(state.get("baseline_intents", []))

        tier = str(state.get("cycle_tier", "FULL"))
        if runtime.settings.enable_llm_supervisor and tier != "SKIP":
            decisions, parse_errors = parse_supervisor_decisions(state.get("supervisor_message", ""))
            for problem in parse_errors:
                LOGGER.warning("decision parse issue run_id=%s: %s", run_id, problem)
                errors.append(problem)
            if not decisions:
                # The supervisor can produce a correct read but end in prose with no
                # JSON block. Rather than discard the analysis as all-WAIT, fall back
                # to the deterministic demand-zone laddered proposals so a real, gated
                # bid is still placed where the math found support. Symbols with no
                # baseline zone still WAIT.
                fallback = decisions_from_intents(baseline) if baseline else []
                if fallback:
                    LOGGER.info(
                        "supervisor emitted no decision JSON run_id=%s; falling back to "
                        "%s deterministic demand-zone proposal(s)",
                        run_id,
                        len(fallback),
                    )
                    covered = {decision.symbol for decision in fallback}
                    decisions = fallback + [
                        wait_decision(symbol, "no supervisor decision; no baseline zone proposal")
                        for symbol in symbols
                        if symbol not in covered
                    ]
                else:
                    # Fail safe: no valid decision and no baseline means WAIT everywhere.
                    decisions = [
                        wait_decision(symbol, "; ".join(parse_errors) or "no decision returned")
                        for symbol in symbols
                    ]
        else:
            # SKIP tier (no material change) or deterministic mode: take the cheap
            # baseline path with no LLM tokens. A SKIP only happens when the
            # baseline proposed nothing, so this resolves to all-WAIT.
            decisions = decisions_from_intents(baseline)
            if not decisions:
                reason = (
                    "no material change this cycle (cost tier=SKIP)"
                    if tier == "SKIP"
                    else "deterministic strategy emitted no proposal"
                )
                decisions = [
                    wait_decision(symbol, reason, source="deterministic") for symbol in symbols
                ]

        for decision in decisions:
            runtime.store.save_supervisor_decision(
                decision_id=decision.id,
                run_id=run_id,
                action=decision.action.value,
                symbol=decision.symbol,
                payload=decision.summary_payload(),
                source=decision.source,
            )
            runtime._emit(
                "decision",
                "supervisor",
                {
                    "action": decision.action.value,
                    "symbol": decision.symbol,
                    "confidence": decision.confidence,
                    "rationale": decision.rationale[:200],
                },
            )
            LOGGER.info(
                "supervisor decision run_id=%s decision_id=%s action=%s symbol=%s confidence=%.4f rationale=%s",
                run_id,
                decision.id,
                decision.action.value,
                decision.symbol,
                decision.confidence,
                decision.rationale,
            )
        return {"decisions": decisions, "errors": errors}

    # ------------------------------------------------------------------ #
    # risk_gate

    def risk_gate(self, state: RuntimeGraphState) -> RuntimeGraphState:
        runtime = self.runtime
        run_id = str(state["run_id"])
        snapshots = dict(state.get("snapshots", {}))
        level_maps = dict(state.get("level_maps", {}))
        evidence = list(state.get("evidence", []))
        evidence_by_id = {record.id: record for record in evidence}
        approved: list[tuple[SupervisorDecision, Any, Any]] = []
        approved_count = 0
        rejected_count = 0
        open_orders = {order.id: order for order in runtime.store.open_positions()}
        open_position_symbols = {order.symbol for order in open_orders.values()}
        reserved_quote_notional: dict[str, float] = {}

        for decision in state.get("decisions", []):
            if decision.action == DecisionAction.WAIT:
                continue
            if decision.action == DecisionAction.BUY:
                outcome = self._gate_buy(
                    decision,
                    run_id,
                    snapshots,
                    evidence_by_id,
                    open_position_symbols,
                    reserved_quote_notional,
                    level_maps,
                )
                if outcome is None:
                    rejected_count += 1
                    continue
                intent, proposal = outcome
                approved_count += 1
                approved.append((decision, intent, proposal))
                open_position_symbols.add(decision.symbol)
                _, quote_asset = split_symbol(proposal.symbol)
                reserved_quote_notional[quote_asset] = (
                    reserved_quote_notional.get(quote_asset, 0.0) + proposal.notional_usd
                )
                continue
            # CLOSE / SELL / ADJUST target an existing open order.
            target = open_orders.get(decision.target_order_id or "")
            reasons: list[str] = []
            if target is None:
                reasons.append(f"target order {decision.target_order_id} is not an open order")
            elif decision.action == DecisionAction.ADJUST:
                new_tp = decision.new_take_profit_price or target.take_profit_price
                new_sl = decision.new_stop_loss_price or target.stop_loss_price
                if new_sl >= new_tp:
                    reasons.append(f"ADJUST rejected: stop loss {new_sl} must stay below take profit {new_tp}")
            elif decision.action in {DecisionAction.CLOSE, DecisionAction.SELL}:
                # Min-hold: the agent team cannot close a filled position within the
                # cooldown. The exit ladder (via exchange_sync, not this gate) fires
                # anytime, and operator-sourced closes bypass this.
                if (
                    target.status == OrderStatus.POSITION_OPEN
                    and not decision.source.startswith("operator")
                    and runtime.config.risk.min_hold_hours > 0
                ):
                    age_minutes = _order_age_minutes(target.opened_at)
                    min_hold_minutes = runtime.config.risk.min_hold_hours * 60
                    if age_minutes is not None and age_minutes < min_hold_minutes:
                        reasons.append(
                            f"min-hold {runtime.config.risk.min_hold_hours:g}h not elapsed "
                            f"(position {age_minutes:.0f}m old); deterministic stop still protects"
                        )
            if reasons:
                rejected_count += 1
                runtime.store.save_supervisor_decision(
                    decision_id=decision.id,
                    run_id=run_id,
                    action=decision.action.value,
                    symbol=decision.symbol,
                    payload=decision.summary_payload(),
                    gate_approved=False,
                    gate_reasons=reasons,
                    source=decision.source,
                )
                runtime._emit(
                    "risk_decision",
                    "pretrade_check",
                    {"symbol": decision.symbol, "approved": False, "reasons": reasons},
                )
                LOGGER.info(
                    "decision blocked run_id=%s decision_id=%s action=%s reasons=%s",
                    run_id,
                    decision.id,
                    decision.action.value,
                    "; ".join(reasons),
                )
                continue
            approved_count += 1
            approved.append((decision, None, None))

        return {
            "approved": approved,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
        }

    def _gate_buy(
        self,
        decision: SupervisorDecision,
        run_id: str,
        snapshots: dict[str, Any],
        evidence_by_id: dict[str, EvidenceRecord],
        open_position_symbols: set[str],
        reserved_quote_notional: dict[str, float] | None = None,
        level_maps: dict[str, Any] | None = None,
    ) -> tuple[Any, Any] | None:
        """Deterministic gate for BUY decisions; returns (intent, proposal) or None."""
        runtime = self.runtime
        intent = decision.to_intent(runtime.config)
        level_maps = level_maps or {}

        def reject(reasons: list[str], gate: str) -> None:
            runtime.store.save_supervisor_decision(
                decision_id=decision.id,
                run_id=run_id,
                action=decision.action.value,
                symbol=decision.symbol,
                payload=decision.summary_payload(),
                gate_approved=False,
                gate_reasons=reasons,
                source=decision.source,
            )
            runtime._emit(
                "risk_decision", gate, {"symbol": decision.symbol, "approved": False, "reasons": reasons}
            )

        # Hard maker discipline: take the lower of the LLM price and the
        # maker-pullback price, so a chase is pulled back to the maker price while
        # a deeper support bid is preserved. The original price is kept for audit.
        gate_snapshot = snapshots.get(intent.symbol)
        if (
            runtime.config.risk.hard_maker_entry
            and gate_snapshot is not None
            and gate_snapshot.bid_price > 0
        ):
            suggested = intent.limit_price
            maker_price = maker_pullback_price(gate_snapshot, runtime.config.risk)
            intent.limit_price = maker_price if suggested <= 0 else min(suggested, maker_price)
            if abs(suggested - intent.limit_price) > 1e-9:
                runtime.store.log_event(
                    "entry_price_overridden_maker",
                    {
                        "decision_id": decision.id,
                        "symbol": intent.symbol,
                        "llm_price": suggested,
                        "maker_price": maker_price,
                        "applied_price": intent.limit_price,
                        "honored_support_bid": suggested > 0 and suggested < maker_price,
                        "bid": gate_snapshot.bid_price,
                    },
                )
        # Geometric edge: the edge that matters for a limit-at-support entry is the
        # distance from THIS entry to the target (resistance), not a sentiment score.
        # The supervisor's self-reported expected_edge_bps under-counts deep-support
        # asymmetry; recompute from the actual entry->target geometry and use the
        # larger value for sizing and the gate.
        entry_price = intent.limit_price
        if entry_price > 0:
            target_price = intent.target_price
            if not target_price or target_price <= entry_price:
                target_price = entry_price * (1 + max(0.0, intent.take_profit_pct))
            geometric_edge_bps = (target_price - entry_price) / entry_price * 10_000
            if geometric_edge_bps > intent.expected_edge_bps:
                intent.expected_edge_bps = round(geometric_edge_bps, 4)
        # Conviction-scaled sizing overrides the LLM/baseline quantity so both
        # price and size are deterministic; a sub-minimum size rejects the BUY
        # instead of sending an unfillable order. The suggested quantity is kept
        # for audit. risk_review's size_multiplier is binding: the tightest
        # multiplier across its consultations wins, and 0 vetoes outright.
        risk_review_mult = 1.0
        for consultation in decision.consultations:
            if consultation.agent == "risk_review":
                risk_review_mult = min(risk_review_mult, consultation.size_multiplier)
        if risk_review_mult <= 0.0:
            reasons = ["risk_review vetoed the trade (size_multiplier=0)"]
            LOGGER.info(
                "risk_review veto run_id=%s decision_id=%s symbol=%s",
                run_id,
                decision.id,
                decision.symbol,
            )
            reject(reasons, "risk_review")
            return None
        # Data-quality haircut: evidence dominated by degraded fallback sources
        # (web news, coarse proxies) shrinks the size proportionally.
        symbol_live_evidence = [
            record
            for record in evidence_by_id.values()
            if record.symbol == intent.symbol and not record.is_placeholder
        ]
        data_quality = (
            sum(record.quality for record in symbol_live_evidence) / len(symbol_live_evidence)
            if symbol_live_evidence
            else 1.0
        )
        if intent.limit_price > 0:
            atr_pct = None
            if gate_snapshot is not None and gate_snapshot.last_price > 0 and gate_snapshot.atr:
                atr_pct = gate_snapshot.atr / gate_snapshot.last_price
            suggested_qty = intent.quantity
            target_notional = conviction_size(
                confidence=intent.confidence,
                expected_edge_bps=intent.expected_edge_bps,
                atr_pct=atr_pct,
                budget_usd=runtime.config.live.capital_budget_usd,
                sizing=runtime.config.sizing,
                # Both are [0,1] discretionary haircuts on the sized notional:
                # the binding risk_review lever and the data-quality penalty.
                quality_mult=risk_review_mult * data_quality,
            )
            if target_notional <= 0:
                reasons = [
                    "conviction-scaled size below exchange minimum notional "
                    f"({runtime.config.sizing.min_notional_usd:g} USD); setup too marginal to size"
                ]
                LOGGER.info(
                    "sizing rejected decision run_id=%s decision_id=%s symbol=%s confidence=%.4f edge_bps=%.2f",
                    run_id,
                    decision.id,
                    decision.symbol,
                    intent.confidence,
                    intent.expected_edge_bps,
                )
                reject(reasons, "sizing")
                return None
            intent.quantity = round(target_notional / intent.limit_price, 8)
            if abs(suggested_qty - intent.quantity) > 1e-12:
                runtime.store.log_event(
                    "entry_size_scaled",
                    {
                        "decision_id": decision.id,
                        "symbol": intent.symbol,
                        "llm_quantity": suggested_qty,
                        "sized_quantity": intent.quantity,
                        "target_notional_usd": target_notional,
                        "confidence": intent.confidence,
                        "expected_edge_bps": intent.expected_edge_bps,
                        "atr_pct": atr_pct,
                    },
                )
        runtime.store.save_trade_intent(intent, run_id=run_id)
        proposal = intent.to_proposal()
        runtime.store.save_proposal(proposal)
        # Bind the decision to the evidence it actually cited. Resolved refs are
        # the records gathered this cycle whose ids the proposal names; anything
        # else is unresolved (fabricated or hallucinated by the supervisor).
        cited = list(proposal.evidence_ids)
        proposal_evidence = [evidence_by_id[item] for item in cited if item in evidence_by_id]
        unresolved_refs = [item for item in cited if item not in evidence_by_id]
        if unresolved_refs:
            # Fabricated citations are an audit hazard even when we do not block,
            # so record them whenever they appear.
            runtime.store.log_event(
                "evidence_refs_unresolved",
                {
                    "decision_id": decision.id,
                    "symbol": decision.symbol,
                    "source": decision.source,
                    "unresolved": unresolved_refs,
                    "resolved": [record.id for record in proposal_evidence],
                },
            )
        if not proposal_evidence:
            # Nothing cited resolved. Fall back to this cycle's live deterministic
            # evidence for the symbol so freshness checks still apply. Placeholder
            # records are excluded exactly as StrategyAgent excludes them.
            proposal_evidence = [
                record
                for record in evidence_by_id.values()
                if record.symbol == decision.symbol and not record.is_placeholder
            ]

        # Fail closed on fabricated citations: an LLM BUY whose evidence_refs
        # resolve to nothing has no auditable basis. The deterministic path
        # (StrategyAgent) always cites real ids, so it is unaffected.
        cited_anything = bool(cited)
        none_resolved = not any(item in evidence_by_id for item in cited)
        if (
            runtime.config.risk.require_evidence_refs
            and decision.source != "deterministic"
            and cited_anything
            and none_resolved
        ):
            reasons = [
                "evidence_refs cite no evidence gathered this cycle: "
                + ", ".join(unresolved_refs)
            ]
            LOGGER.info(
                "evidence binding blocked decision run_id=%s decision_id=%s symbol=%s unresolved=%s",
                run_id,
                decision.id,
                decision.symbol,
                ", ".join(unresolved_refs),
            )
            reject(reasons, "evidence_binding")
            return None

        # Zone-anchored bids may rest well below market (the next demand zone,
        # typically 3-6% down); an un-anchored bid keeps the tight leash.
        risk_cfg = runtime.config.risk
        if risk_cfg.require_zone_anchored_bids and self._is_zone_anchored(
            intent, level_maps.get(intent.symbol)
        ):
            max_deviation_pct = risk_cfg.max_bid_depth_pct
        else:
            max_deviation_pct = risk_cfg.unanchored_max_deviation_pct
        blockers = self._pretrade_blockers(
            intent, snapshots.get(intent.symbol), open_position_symbols, max_deviation_pct
        )
        if blockers:
            LOGGER.info(
                "pre-trade check blocked decision run_id=%s decision_id=%s symbol=%s reasons=%s",
                run_id,
                decision.id,
                decision.symbol,
                "; ".join(blockers),
            )
            runtime.store.log_event(
                "pretrade_blocked",
                {"intent_id": intent.id, "symbol": intent.symbol, "reasons": blockers},
            )
            reject(blockers, "pretrade_check")
            return None
        if runtime.store.get_setting("kill_switch", False):
            LOGGER.warning(
                "kill switch blocks decision run_id=%s decision_id=%s symbol=%s",
                run_id,
                decision.id,
                decision.symbol,
            )
            runtime.store.log_event(
                "agent_kill_switch_before_intent",
                {"intent_id": intent.id, "symbol": intent.symbol},
            )
        available_quote_balance = runtime.available_quote_balance(intent.symbol)
        reserved_notional = 0.0
        if reserved_quote_notional:
            _, quote_asset = split_symbol(intent.symbol)
            reserved_notional = reserved_quote_notional.get(quote_asset, 0.0)
            if available_quote_balance is not None:
                available_quote_balance = max(0.0, available_quote_balance - reserved_notional)
        risk_state = runtime.risk.runtime_state(
            runtime.store,
            runtime.config.mode,
            available_quote_balance_usd=available_quote_balance,
        )
        risk_state.open_notional_usd += reserved_notional
        risk_decision = runtime.risk.evaluate(proposal, proposal_evidence, risk_state, runtime.config)
        runtime.store.save_risk_decision(risk_decision)
        runtime._emit(
            "risk_decision",
            "risk_governor",
            {
                "symbol": intent.symbol,
                "approved": risk_decision.approved,
                "reasons": risk_decision.reasons,
            },
        )
        if not risk_decision.approved:
            LOGGER.info(
                "risk rejected decision run_id=%s decision_id=%s symbol=%s reasons=%s",
                run_id,
                decision.id,
                decision.symbol,
                "; ".join(risk_decision.reasons),
            )
            runtime.store.log_event(
                "risk_rejection",
                {"intent_id": intent.id, "proposal_id": proposal.id, "reasons": risk_decision.reasons},
            )
            reject(risk_decision.reasons, "risk_governor")
            return None
        LOGGER.info(
            "risk approved decision run_id=%s decision_id=%s proposal_id=%s symbol=%s open_positions=%s",
            run_id,
            decision.id,
            proposal.id,
            intent.symbol,
            risk_state.open_position_count,
        )
        runtime.store.log_event(
            "risk_approval",
            {
                "intent_id": intent.id,
                "proposal_id": proposal.id,
                "symbol": intent.symbol,
                "open_positions": risk_state.open_position_count,
            },
        )
        return intent, proposal

    # ------------------------------------------------------------------ #
    # execute

    def execute(self, state: RuntimeGraphState) -> RuntimeGraphState:
        runtime = self.runtime
        run_id = str(state["run_id"])
        snapshots = dict(state.get("snapshots", {}))
        errors = list(state.get("errors", []))
        submitted: list[str] = []
        for decision, intent, proposal in state.get("approved", []):
            try:
                if decision.action == DecisionAction.BUY:
                    order = self._execute_buy(decision, intent, proposal, run_id)
                    if order is not None:
                        submitted.append(order.id)
                elif decision.action in {DecisionAction.CLOSE, DecisionAction.SELL}:
                    self._execute_close(decision, snapshots, run_id)
                elif decision.action == DecisionAction.ADJUST:
                    self._execute_adjust(decision, run_id)
            except Exception as exc:
                errors.append(str(exc))
                LOGGER.exception(
                    "decision execution failed run_id=%s decision_id=%s action=%s symbol=%s",
                    run_id,
                    decision.id,
                    decision.action.value,
                    decision.symbol,
                )
                runtime.store.log_event(
                    "decision_execution_error",
                    {
                        "decision_id": decision.id,
                        "action": decision.action.value,
                        "symbol": decision.symbol,
                        "error": str(exc),
                    },
                )
        return {"submitted_orders": submitted, "errors": errors}

    def _execute_buy(
        self, decision: SupervisorDecision, intent: Any, proposal: Any, run_id: str
    ) -> OrderRecord | None:
        runtime = self.runtime
        mode = runtime.settings.trading_agent_execution_mode
        if mode in {"testnet", "live"}:
            order = runtime._submit_exchange_limit_entry(proposal, intent, run_id)
        else:
            LOGGER.warning(
                "execution mode blocked run_id=%s decision_id=%s execution_mode=%s",
                run_id,
                decision.id,
                mode,
            )
            runtime.store.log_event(
                "execution_mode_blocked",
                {"intent_id": intent.id, "proposal_id": proposal.id, "execution_mode": mode},
            )
            return None
        order.decision_id = decision.id
        runtime.store.save_order(order)
        runtime.store.save_supervisor_decision(
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
        runtime._emit(
            "order",
            "execution",
            {"mode": order.mode, "symbol": order.symbol, "price": order.price, "quantity": order.quantity},
        )
        LOGGER.info(
            "%s order submitted run_id=%s order_id=%s symbol=%s price=%s quantity=%s",
            order.mode,
            run_id,
            order.id,
            order.symbol,
            order.price,
            order.quantity,
        )
        runtime.store.log_event(
            f"{order.mode}_spot_entry_submitted",
            {
                "order_id": order.id,
                "decision_id": decision.id,
                "proposal_id": proposal.id,
                "symbol": order.symbol,
                "price": order.price,
                "quantity": order.quantity,
                "take_profit_price": order.take_profit_price,
                "stop_loss_price": order.stop_loss_price,
            },
        )
        return order

    def _execute_close(
        self, decision: SupervisorDecision, snapshots: dict[str, Any], run_id: str
    ) -> None:
        runtime = self.runtime
        try:
            order = self._open_order(decision.target_order_id)
        except ValueError:
            # Stale target (already closed/canceled, e.g. the agent recalled an
            # old order id from memory). Benign no-op, not a cycle error.
            LOGGER.info(
                "close no-op: target order %s is not open run_id=%s decision_id=%s",
                decision.target_order_id,
                run_id,
                decision.id,
            )
            runtime.store.log_event(
                "close_noop",
                {"order_id": decision.target_order_id, "decision_id": decision.id, "reason": "target not open"},
            )
            return
        reason = f"{decision.source.upper()}_CLOSE"
        if order.mode not in {"testnet", "live"}:
            LOGGER.warning(
                "close skipped: non-exchange order cannot be closed on the venue "
                "run_id=%s decision_id=%s order_id=%s mode=%s",
                run_id,
                decision.id,
                order.id,
                order.mode,
            )
            runtime.store.log_event(
                "close_skipped_non_exchange_order",
                {"order_id": order.id, "decision_id": decision.id, "mode": order.mode},
            )
            return
        runtime.exchange_sync.close_position(order, price=decision.limit_price, reason=reason)
        runtime.store.save_supervisor_decision(
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
        runtime._emit(
            "order",
            "execution",
            {"mode": order.mode, "symbol": order.symbol, "close": True, "reason": reason},
        )
        LOGGER.info(
            "close executed run_id=%s decision_id=%s order_id=%s status=%s",
            run_id,
            decision.id,
            order.id,
            order.status.value,
        )

    def _execute_adjust(self, decision: SupervisorDecision, run_id: str) -> None:
        runtime = self.runtime
        order = self._open_order(decision.target_order_id)
        before = {"take_profit_price": order.take_profit_price, "stop_loss_price": order.stop_loss_price}
        if decision.new_take_profit_price is not None:
            order.take_profit_price = round(float(decision.new_take_profit_price), 8)
        if decision.new_stop_loss_price is not None:
            order.stop_loss_price = round(float(decision.new_stop_loss_price), 8)
            # Sync the tiered ladder's live stop so the operator/agent ADJUST is
            # not overwritten by the next reconcile (the ladder owns the stop).
            if order.exit_plan is not None:
                order.exit_plan.current_stop_price = order.stop_loss_price
        runtime.store.save_order(order)
        runtime.store.save_supervisor_decision(
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
        runtime.store.log_event(
            "order_bracket_adjusted",
            {
                "order_id": order.id,
                "decision_id": decision.id,
                "before": before,
                "after": {
                    "take_profit_price": order.take_profit_price,
                    "stop_loss_price": order.stop_loss_price,
                },
            },
        )
        runtime._emit(
            "order",
            "execution",
            {
                "mode": order.mode,
                "symbol": order.symbol,
                "adjust": True,
                "take_profit_price": order.take_profit_price,
                "stop_loss_price": order.stop_loss_price,
            },
        )
        LOGGER.info(
            "bracket adjusted run_id=%s decision_id=%s order_id=%s tp=%s sl=%s",
            run_id,
            decision.id,
            order.id,
            order.take_profit_price,
            order.stop_loss_price,
        )

    def _open_order(self, order_id: str | None) -> OrderRecord:
        for order in self.runtime.store.open_positions():
            if order.id == order_id:
                return order
        raise ValueError(f"order {order_id} is not open")

    # ------------------------------------------------------------------ #
    # report

    def report(self, state: RuntimeGraphState) -> RuntimeGraphState:
        runtime = self.runtime
        run_id = str(state["run_id"])
        thread_id = str(state["thread_id"])
        cycle = int(state["cycle"])
        errors = list(state.get("errors", []))
        decisions = list(state.get("decisions", []))
        wait_count = sum(1 for d in decisions if d.action == DecisionAction.WAIT)
        result = AgentRunResult(
            run_id=run_id,
            thread_id=thread_id,
            cycle=cycle,
            evidence_count=len(state.get("evidence", [])),
            intent_count=len(state.get("baseline_intents", [])),
            approved_count=int(state.get("approved_count", 0)),
            rejected_count=int(state.get("rejected_count", 0)),
            submitted_trades=len(state.get("submitted_orders", [])),
            reconciled_trades=int(state.get("reconciled_count", 0)),
            mcp_tool_count=int(state.get("mcp_tool_count", 0)),
            error_count=len(errors),
            decision_count=len(decisions),
            wait_count=wait_count,
        )
        runtime.store.finish_agent_run(run_id, "completed")
        runtime.store.set_setting(
            "agent_checkpoint",
            {"thread_id": thread_id, "latest_run_id": run_id, "last_cycle": cycle},
        )
        runtime.store.set_heartbeat(
            {
                "status": "completed",
                "thread_id": thread_id,
                "run_id": run_id,
                "cycle": cycle,
                "error_count": len(errors),
            }
        )
        LOGGER.info(
            "cycle completed run_id=%s evidence=%s decisions=%s approved=%s rejected=%s submitted=%s errors=%s",
            run_id,
            result.evidence_count,
            result.decision_count,
            result.approved_count,
            result.rejected_count,
            result.submitted_trades,
            len(errors),
        )
        self.active_run_id = None
        return {"result": asdict(result)}

    # ------------------------------------------------------------------ #
    # shared helpers

    def _record_token_usage(
        self, run_id: str, cycle: int, usage_by_model: dict[str, Any]
    ) -> None:
        """Log this cycle's LLM token spend and maintain a running total.

        Never raises into the cycle: cost accounting must not break trading.
        """
        try:
            summary = summarize_usage(usage_by_model)
            if summary["total_tokens"] == 0:
                return
            store = self.runtime.store
            previous = store.get_setting("token_usage_cumulative", {}) or {}
            cumulative = {
                "cycles": int(previous.get("cycles", 0)) + 1,
                "input_tokens": int(previous.get("input_tokens", 0)) + summary["input_tokens"],
                "output_tokens": int(previous.get("output_tokens", 0)) + summary["output_tokens"],
                "cost_usd": round(float(previous.get("cost_usd", 0.0)) + summary["cost_usd"], 6),
                "since": previous.get("since") or utc_iso(),
                "fully_priced": bool(previous.get("fully_priced", True)) and summary["fully_priced"],
            }
            store.set_setting("token_usage_cumulative", cumulative)
            store.log_event(
                "token_usage",
                {"run_id": run_id, "cycle": cycle, "cycle_usage": summary, "cumulative": cumulative},
            )
            LOGGER.info(
                "token usage run_id=%s cycle=%s input=%s cached=%s output=%s reasoning=%s "
                "cost_usd=%.4f%s | cumulative cycles=%s cost_usd=%.4f",
                run_id,
                cycle,
                summary["input_tokens"],
                summary["cached_tokens"],
                summary["output_tokens"],
                summary["reasoning_tokens"],
                summary["cost_usd"],
                "" if summary["fully_priced"] else " (partial: unpriced model present)",
                cumulative["cycles"],
                cumulative["cost_usd"],
            )
        except Exception:  # accounting must never break a trading cycle
            LOGGER.exception("token usage accounting failed run_id=%s cycle=%s", run_id, cycle)

    @staticmethod
    def _is_zone_anchored(intent: TradeIntent, level_map: Any) -> bool:
        """True when the bid rests inside a computed support (demand) zone.

        Either the intent names a zone id that resolves, or its limit price falls
        within a support zone's band (with a small tolerance). Only anchored bids
        are allowed to rest deep below market.
        """
        if level_map is None or intent.limit_price <= 0:
            return False
        if intent.zone_id:
            zone = level_map.zone_by_id(intent.zone_id)
            if zone is not None and zone.side == "support":
                return True
        tol = 0.003
        for zone in level_map.support_zones:
            if zone.low * (1 - tol) <= intent.limit_price <= zone.high * (1 + tol):
                return True
        return False

    def _pretrade_blockers(
        self,
        intent: TradeIntent,
        snapshot: Any,
        open_position_symbols: set[str],
        max_deviation_pct: float = 0.02,
    ) -> list[str]:
        """Deterministic relevance checks before any order leaves the system.

        1. No doubling up: skip symbols that already have an open position.
        2. Price sanity: the limit price must stay within ``max_deviation_pct`` of
           the live market price. Zone-anchored support bids get a wider band
           (max_bid_depth_pct) so they can rest at the real demand zone; un-anchored
           bids keep the tight default.
        3. Maker-pullback discipline: a BUY limit must rest at/below the bid so it
           fills on a dip as a maker order, never chase. A small tolerance above
           the bid (max_cross_spread_bps) only absorbs research->gate drift; any
           more is refused.
        """
        blockers: list[str] = []
        if intent.symbol in open_position_symbols:
            blockers.append("position already open for symbol; not adding to it")
        if snapshot is None:
            blockers.append("no market snapshot available to verify limit price")
            return blockers
        if snapshot.last_price > 0:
            deviation = abs(intent.limit_price - snapshot.last_price) / snapshot.last_price
            if deviation > max_deviation_pct:
                blockers.append(
                    f"limit price {intent.limit_price} deviates {deviation:.2%} from "
                    f"market {snapshot.last_price} (max {max_deviation_pct:.0%})"
                )
        if intent.side.value == "BUY" and snapshot.bid_price > 0:
            max_cross_bps = self.runtime.config.risk.max_cross_spread_bps
            ceiling = snapshot.bid_price * (1 + max_cross_bps / 10_000)
            if intent.limit_price > ceiling:
                over_bps = (intent.limit_price / snapshot.bid_price - 1) * 10_000
                blockers.append(
                    f"BUY limit {intent.limit_price} is {over_bps:.1f} bps above bid "
                    f"{snapshot.bid_price}; maker-pullback entries must rest at/below the "
                    f"bid (max cross {max_cross_bps:.0f} bps), not chase"
                )
        return blockers

    def _strategy_wait_diagnostics(self, symbol: str, records: list[EvidenceRecord]) -> dict[str, Any]:
        # Mirror StrategyAgent: placeholder evidence never moves decisions.
        records = [record for record in records if not record.is_placeholder]
        config = self.runtime.config
        if not records:
            return {
                "reason": "no live evidence (placeholder-only sources)",
                "expected_edge_bps": None,
                "min_expected_edge_bps": config.risk.min_expected_edge_bps,
                "avg_confidence": None,
            }
        tuning = config.strategy
        weighted_score = 0.0
        weight_total = 0.0
        confidence_total = 0.0
        for record in records:
            weight = tuning.agent_weights.get(record.agent, tuning.default_agent_weight)
            weighted_score += record.score * weight
            weight_total += weight
            confidence_total += record.confidence
        combined_score = weighted_score / weight_total if weight_total else 0.0
        expected_edge_bps = combined_score * tuning.edge_scale_bps
        avg_confidence = confidence_total / len(records)
        if expected_edge_bps < config.risk.min_expected_edge_bps:
            reason = "expected edge below configured threshold"
        else:
            reason = "strategy did not emit a proposal"
        return {
            "reason": reason,
            "expected_edge_bps": round(expected_edge_bps, 4),
            "min_expected_edge_bps": config.risk.min_expected_edge_bps,
            "avg_confidence": round(avg_confidence, 6),
        }

    def _log_prompt_registry(self, run_id: str, cycle: int, symbols: list[str]) -> None:
        input_summary = f"cycle={cycle}; symbols={','.join(symbols)}"
        for prompt in all_prompts():
            self.runtime.store.save_prompt_log(
                run_id=run_id,
                agent_name=prompt.name,
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                input_summary=input_summary,
                output_summary="registered for supervised cycle",
            )

    def _load_mcp_tools(self, run_id: str) -> MCPToolLoadResult:
        runtime = self.runtime
        enabled = sum(1 for server in runtime.mcp_loader.servers if server.enabled)
        LOGGER.info(
            "loading MCP tools run_id=%s servers_configured=%s servers_enabled=%s",
            run_id,
            len(runtime.mcp_loader.servers),
            enabled,
        )
        try:
            result = runtime.mcp_loader.load_tools_sync()
        except Exception as exc:
            result = MCPToolLoadResult(tools=[], errors=[str(exc)], server_count=0)
            LOGGER.exception("MCP tool loader raised run_id=%s", run_id)
        runtime.store.save_tool_call_log(
            run_id=run_id,
            agent_name="supervisor",
            tool_name="mcp_loader",
            input_summary=(
                f"servers_configured={len(runtime.mcp_loader.servers)}; "
                f"servers_enabled={enabled}; hosted_only=True"
            ),
            output_summary=f"tools={len(result.tools)}; servers={result.server_count}",
            error="; ".join(result.errors) if result.errors else None,
        )
        LOGGER.info(
            "MCP tools loaded run_id=%s tools=%s servers=%s blocked=%s filtered=%s errors=%s",
            run_id,
            len(result.tools),
            result.server_count,
            result.blocked_tool_count,
            result.filtered_tool_count,
            len(result.errors),
        )
        return result
