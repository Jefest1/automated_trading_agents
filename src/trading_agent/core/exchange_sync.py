"""Reconcile local order records against live exchange state (testnet/live).

The local `orders` table is only a cache: every cycle (and on demand from the
REPL/CLI) this module pulls the authoritative order status and fills from the
exchange, ingests them idempotently, manages the virtual TP/SL bracket for
open positions, and computes per-trade realized PnL from actual fills.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trading_agent.core.config import ExitConfig, Settings
from trading_agent.core.exit_ladder import (
    apply_tier_fill,
    remaining_quantity,
    stop_reason,
    update_trail,
)
from trading_agent.core.logging import get_logger
from trading_agent.core.models import (
    ExitLeg,
    FillRecord,
    MarketSnapshot,
    OrderRecord,
    OrderStatus,
    Side,
    utc_iso,
)
from trading_agent.core.pnl import round_trip_pnl, split_symbol
from trading_agent.core.reflection import build_reflection
from trading_agent.core.storage import Store
from trading_agent.exchange import BinanceCredentials, BinanceSpotAdapter

LOGGER = get_logger("exchange_sync")

# Exchange order states that mean "no more fills are coming".
_TERMINAL_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH"}
_SYNC_MODES = {"testnet", "live"}
# How long a PENDING_SUBMIT row may sit unconfirmed before the reconciler
# treats the submit as failed (the venue was queried and knows nothing).
_PENDING_SUBMIT_GRACE_SECONDS = 60.0


def _age_seconds(stamp: str) -> float:
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds()


class ExchangeReconciler:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        *,
        adapter: BinanceSpotAdapter | None = None,
        credentials: BinanceCredentials | None = None,
        exit_config: ExitConfig | None = None,
        bid_ttl_minutes: int = 0,
    ) -> None:
        self.store = store
        self.settings = settings
        self._adapter = adapter
        self._credentials = credentials
        self.exit_config = exit_config or ExitConfig()
        # Auto-expire a resting demand-zone bid after this many minutes (GTD-style)
        # so a stale bid never fills days later into a dead thesis. 0 disables.
        self.bid_ttl_minutes = bid_ttl_minutes

    @property
    def available(self) -> bool:
        if self._adapter is not None and self._credentials is not None:
            return True
        return self.settings.binance_api_key is not None and self.settings.binance_api_secret is not None

    @property
    def adapter(self) -> BinanceSpotAdapter:
        if self._adapter is None:
            self._adapter = BinanceSpotAdapter(
                base_url=self.settings.exchange_base_url(), settings=self.settings
            )
        return self._adapter

    @property
    def credentials(self) -> BinanceCredentials:
        if self._credentials is None:
            self._credentials = BinanceSpotAdapter.credentials_from_env(settings=self.settings)
        return self._credentials

    def reconcile(self, snapshots: dict[str, MarketSnapshot] | None = None) -> list[OrderRecord]:
        """Sync every open exchange order; manage TP/SL exits. Returns updated orders."""
        if not self.available:
            return []
        updated: list[OrderRecord] = []
        for order in self.store.open_positions():
            if order.mode not in _SYNC_MODES:
                continue
            try:
                if order.status == OrderStatus.PENDING_SUBMIT:
                    if self._resolve_pending_submit(order):
                        updated.append(order)
                    continue
                changed = self.sync_order(order)
                snapshot = (snapshots or {}).get(order.symbol)
                # Auto-expire an unfilled resting bid past its TTL: a demand-zone bid
                # that has not filled in this window is cancelled so capital is freed
                # and a stale bid never fills later into a changed market.
                if (
                    order.status == OrderStatus.ENTRY_OPEN
                    and order.executed_qty == 0
                    and self.bid_ttl_minutes > 0
                    and _age_seconds(order.opened_at) > self.bid_ttl_minutes * 60
                ):
                    self.cancel_entry(order, reason="BID_TTL_EXPIRED")
                    self.store.log_event(
                        "bid_ttl_expired",
                        {
                            "order_id": order.id,
                            "symbol": order.symbol,
                            "age_minutes": round(_age_seconds(order.opened_at) / 60, 1),
                        },
                    )
                    updated.append(order)
                    continue
                if order.status == OrderStatus.POSITION_OPEN:
                    # Re-check on fill: if the entry just filled but price is already
                    # below the stop (the demand zone broke through the fill), exit
                    # immediately instead of arming the bracket.
                    if (
                        changed
                        and snapshot is not None
                        and order.exit_order_exchange_id is None
                        and order.stop_loss_price > 0
                        and snapshot.last_price > 0
                        and snapshot.last_price < order.stop_loss_price
                    ):
                        LOGGER.info(
                            "zone invalidated on fill order_id=%s symbol=%s price=%s < stop=%s",
                            order.id,
                            order.symbol,
                            snapshot.last_price,
                            order.stop_loss_price,
                        )
                        self.store.log_event(
                            "zone_invalidated_on_fill",
                            {
                                "order_id": order.id,
                                "symbol": order.symbol,
                                "price": snapshot.last_price,
                                "stop_loss_price": order.stop_loss_price,
                            },
                        )
                        self.close_position(order, reason="ZONE_INVALIDATED_ON_FILL")
                        updated.append(order)
                        continue
                    exit_changed = self._manage_tiered_exits(order, snapshot)
                    changed = changed or exit_changed
                if changed:
                    updated.append(order)
            except Exception as exc:  # one bad order must not sink the cycle
                LOGGER.warning("reconcile failed order_id=%s symbol=%s: %s", order.id, order.symbol, exc)
                self.store.log_event(
                    "exchange_sync_error",
                    {"order_id": order.id, "symbol": order.symbol, "error": str(exc)},
                )
        return updated

    def _resolve_pending_submit(self, order: OrderRecord) -> bool:
        """Resolve a row persisted before submit whose exchange response was
        never recorded (crash or malformed response mid-submit).

        If the venue knows the client_order_id the order is adopted and synced
        normally; if the venue has never seen it after a grace window, the row
        is closed as a failed submit. Within the grace window the submit may
        still be in flight, so the row is left alone.
        """
        if order.client_order_id is None:
            order.status = OrderStatus.CANCELED
            order.closed_at = utc_iso()
            order.closed_by = "SUBMIT_UNRESOLVABLE"
            self.store.save_order(order)
            return True
        live = self._query_order(order.symbol, order_id=None, client_order_id=order.client_order_id)
        if live is not None:
            order.exchange_order_id = str(live.get("orderId"))
            order.status = OrderStatus.ENTRY_OPEN
            self.store.save_order(order)
            self.store.log_event(
                "exchange_pending_submit_adopted",
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "client_order_id": order.client_order_id,
                    "exchange_order_id": order.exchange_order_id,
                },
            )
            LOGGER.warning(
                "adopted orphaned exchange order order_id=%s symbol=%s client_order_id=%s",
                order.id,
                order.symbol,
                order.client_order_id,
            )
            self.sync_order(order)
            return True
        if _age_seconds(order.opened_at) < _PENDING_SUBMIT_GRACE_SECONDS:
            return False
        order.status = OrderStatus.CANCELED
        order.closed_at = utc_iso()
        order.closed_by = "SUBMIT_FAILED"
        order.last_synced_at = utc_iso()
        self.store.save_order(order)
        self.store.log_event(
            "exchange_pending_submit_discarded",
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "client_order_id": order.client_order_id,
            },
        )
        return True

    def sync_order(self, order: OrderRecord) -> bool:
        """Refresh one order from the exchange. Returns True when anything changed."""
        if order.exchange_order_id is None and order.client_order_id is None:
            return False
        before = (order.status, order.exchange_status, order.executed_qty, order.exit_order_exchange_id)
        live = self._query_order(
            order.symbol, order_id=order.exchange_order_id, client_order_id=order.client_order_id
        )
        if live is None:
            order.exchange_status = "UNKNOWN"
            order.last_synced_at = utc_iso()
            self.store.save_order(order)
            return before[1] != "UNKNOWN"

        order.exchange_order_id = str(live.get("orderId", order.exchange_order_id))
        order.exchange_status = str(live.get("status", order.exchange_status))
        order.executed_qty = float(live.get("executedQty", order.executed_qty) or 0)
        order.cumulative_quote_qty = float(live.get("cummulativeQuoteQty", order.cumulative_quote_qty) or 0)
        if order.executed_qty > 0:
            order.avg_fill_price = round(order.cumulative_quote_qty / order.executed_qty, 8)
            self._ingest_fills(order, is_exit=False)
        order.last_synced_at = utc_iso()

        if order.status == OrderStatus.ENTRY_OPEN:
            if order.exchange_status == "FILLED":
                order.status = OrderStatus.POSITION_OPEN
                self.store.log_event(
                    "exchange_entry_filled",
                    {
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "avg_fill_price": order.avg_fill_price,
                        "executed_qty": order.executed_qty,
                        # Proof of execution: the quote asset actually deducted
                        # (cummulativeQuoteQty) plus the post-fill balances.
                        "quote_spent": order.cumulative_quote_qty,
                        "balances_after_fill": self._balance_proof(order.symbol),
                    },
                )
            elif order.exchange_status in _TERMINAL_STATUSES and order.executed_qty == 0:
                order.status = OrderStatus.CANCELED
                order.closed_at = utc_iso()
                self.store.log_event(
                    "exchange_entry_canceled",
                    {"order_id": order.id, "symbol": order.symbol, "exchange_status": order.exchange_status},
                )

        if order.exit_order_exchange_id is not None or order.exit_client_order_id is not None:
            self._sync_exit(order)

        self.store.save_order(order)
        after = (order.status, order.exchange_status, order.executed_qty, order.exit_order_exchange_id)
        return before != after

    def close_position(
        self,
        order: OrderRecord,
        *,
        price: float | None = None,
        reason: str = "OPERATOR_CLOSE",
    ) -> OrderRecord:
        """Close a live order: cancel an unfilled entry, or sell out an open position."""
        self.sync_order(order)
        if order.status == OrderStatus.ENTRY_OPEN and order.executed_qty == 0:
            return self.cancel_entry(order, reason=reason)
        if order.status not in {OrderStatus.ENTRY_OPEN, OrderStatus.POSITION_OPEN}:
            # Idempotent: the order already closed/canceled (e.g. TP/SL hit, or a
            # stale agent target). Treat as a benign no-op instead of raising.
            LOGGER.info(
                "close_position no-op: order %s already %s", order.id, order.status.value
            )
            self.store.log_event(
                "close_noop",
                {"order_id": order.id, "status": order.status.value, "reason": reason},
            )
            return order
        if order.status == OrderStatus.ENTRY_OPEN:
            # Partially filled entry: stop further fills, then sell what we hold.
            self._cancel_on_exchange(order)
            order.status = OrderStatus.POSITION_OPEN
        # Free any balance locked in resting take-profit tiers before selling out,
        # so we never orphan resting sells or over-commit the close.
        self._cancel_resting_legs(order)
        exit_price = price if price is not None else self._marketable_sell_price(order.symbol)
        self._submit_exit(order, exit_price, reason)
        self.store.save_order(order)
        return order

    def cancel_entry(self, order: OrderRecord, *, reason: str = "OPERATOR_CANCEL") -> OrderRecord:
        self._cancel_on_exchange(order)
        order.status = OrderStatus.CANCELED
        order.closed_at = utc_iso()
        order.closed_by = reason
        order.last_synced_at = utc_iso()
        self.store.save_order(order)
        self.store.log_event(
            "exchange_entry_cancel_requested",
            {"order_id": order.id, "symbol": order.symbol, "reason": reason},
        )
        return order

    def _manage_tiered_exits(self, order: OrderRecord, snapshot: MarketSnapshot | None) -> bool:
        """Manage an open position's exits with REAL resting take-profit orders.

        Take-profit tiers are placed as actual LIMIT SELL orders on the exchange,
        so the venue fills them the instant price touches; no hourly polling gap.
        We confirm tier fills (ratcheting the stop) and run a virtual trailing
        stop; a stop-out cancels the resting tiers and market-sells the rest."""
        if order.side != Side.BUY:
            return False
        plan = order.exit_plan
        if plan is None:
            return False
        # A stop-out / operator close market order is already in flight: let
        # sync_order's _sync_exit settle it; do not touch tiers meanwhile.
        if order.exit_order_exchange_id is not None:
            return False
        changed = self._place_resting_tps(order)
        changed = self._sync_resting_legs(order) or changed
        if order.status == OrderStatus.POSITION_OPEN and snapshot is not None:
            changed = self._check_virtual_stop(order, snapshot) or changed
        # Keep the interim realized PnL correct/up to date for a partially
        # scaled-out position even when no new tier fills this cycle: recompute
        # from current fills (idempotent; writes no reflection unless CLOSED) so a
        # value stored before the proration fix self-heals on the next reconcile.
        if any(fill.is_exit for fill in self.store.fills_for_order(order.id)):
            before = order.realized_pnl
            self._compute_realized_pnl(order)
            if order.realized_pnl != before:
                self.store.save_order(order)
                changed = True
        return changed

    def _place_resting_tps(self, order: OrderRecord) -> bool:
        """Rest each not-yet-placed take-profit tier as a real LIMIT SELL at its
        target. Only the TP tiers rest (never the stop/runner) so the held base is
        never over-committed (TP1+TP2 ≈ 70% of size here)."""
        plan = order.exit_plan
        if plan is None:
            return False
        held = order.executed_qty or order.quantity
        entry = order.avg_fill_price or order.price
        # Never rest more than we currently hold (held minus anything already sold)
        # so resting tiers can never over-commit the balance.
        already_exited = sum(f.qty for f in self.store.fills_for_order(order.id) if f.is_exit)
        available = held - already_exited
        placed = False
        for leg in plan.legs:
            if leg.filled or leg.exit_order_exchange_id is not None or leg.exit_client_order_id is not None:
                continue
            if available <= 0:
                break
            qty = min(held * leg.size_pct, available)
            quantity_q, price_q = self.adapter.quantize_order(order.symbol, qty, leg.target_price)
            if float(quantity_q) <= 0:
                # Slice too small to rest after quantization: fold into the runner.
                apply_tier_fill(plan, leg.tier, entry, self.exit_config, filled_qty=0.0)
                placed = True
                continue
            self._place_leg_order(order, leg, quantity_q, price_q)
            available -= float(quantity_q)
            placed = True
        if placed:
            self.store.save_order(order)
        return placed

    def _place_leg_order(self, order: OrderRecord, leg: ExitLeg, quantity_q: Any, price_q: Any) -> None:
        # Deterministic per-tier client id so a crash mid-submit lets us adopt the
        # resting order instead of stacking a duplicate sell.
        client_order_id = f"tx{order.id.replace('_', '')[:20]}t{leg.tier}"
        existing = self._query_order(order.symbol, order_id=None, client_order_id=client_order_id)
        if existing is not None:
            status = str(existing.get("status"))
            executed = float(existing.get("executedQty", 0) or 0)
            if status not in _TERMINAL_STATUSES or executed > 0:
                # Live order OR a terminal order that actually sold base: adopt it.
                # A TP resting below the market fills instantly as a taker; if the
                # binding was lost (crash, later-leg exception), resubmitting would
                # sell the same tier again. Terminal-with-fills is a fill to ingest,
                # never a slot to refill.
                self._bind_leg_exit(leg, str(existing.get("orderId")), client_order_id)
                self.store.save_order(order)
                if executed > 0 and status in _TERMINAL_STATUSES:
                    self.store.log_event(
                        "exchange_take_profit_adopted",
                        {
                            "order_id": order.id,
                            "symbol": order.symbol,
                            "tier": leg.tier,
                            "exchange_order_id": str(existing.get("orderId")),
                            "status": status,
                            "executed_qty": executed,
                        },
                    )
                return
        response = self.adapter.submit_limit_order(
            self.credentials, order.symbol, "SELL", quantity_q, price_q, client_order_id=client_order_id
        )
        raw = response.get("raw", {})
        self._bind_leg_exit(leg, str(raw.get("orderId", "")) or None, client_order_id)
        # Persist the binding IMMEDIATELY: if a later leg's submit raises, an
        # unsaved binding would orphan this live sell and re-place it next tick.
        self.store.save_order(order)
        balances_after_submit = self._balance_proof(order.symbol)
        LOGGER.info(
            "resting take-profit placed order_id=%s symbol=%s tier=%s price=%s qty=%s exchange_order_id=%s",
            order.id, order.symbol, leg.tier, price_q, quantity_q, leg.exit_order_exchange_id,
        )
        if balances_after_submit is not None:
            LOGGER.info("balances after take-profit submit order_id=%s symbol=%s balances=%s", order.id, order.symbol, balances_after_submit)
        self.store.log_event(
            "exchange_take_profit_rested",
            {"order_id": order.id, "symbol": order.symbol, "tier": leg.tier,
             "price": price_q, "quantity": quantity_q, "exchange_order_id": leg.exit_order_exchange_id,
             "balances_after_submit": balances_after_submit},
        )

    def _sync_resting_legs(self, order: OrderRecord) -> bool:
        """Check each resting tier order; on fill, ratchet the stop and bank
        interim PnL. Re-arm a tier whose order died terminal unfilled. Fully close
        once every tier is filled and no runner remains."""
        plan = order.exit_plan
        if plan is None:
            return False
        entry = order.avg_fill_price or order.price
        changed = False
        for leg in plan.legs:
            if leg.filled or leg.exit_order_exchange_id is None:
                continue
            live = self._query_order(
                order.symbol, order_id=leg.exit_order_exchange_id, client_order_id=leg.exit_client_order_id
            )
            if live is None:
                continue
            status = str(live.get("status", ""))
            executed = float(live.get("executedQty", 0) or 0)
            if executed > 0:
                self._ingest_fills(order, is_exit=True, exchange_order_id=leg.exit_order_exchange_id)
            if status == "FILLED":
                quote = float(live.get("cummulativeQuoteQty", 0) or 0)
                order.exit_price = round(quote / executed, 8) if executed else order.exit_price
                apply_tier_fill(
                    plan, leg.tier, entry, self.exit_config, filled_qty=executed,
                    exit_order_exchange_id=leg.exit_order_exchange_id,
                    exit_client_order_id=leg.exit_client_order_id,
                )
                order.stop_loss_price = plan.current_stop_price
                self._compute_realized_pnl(order)
                self.store.log_event(
                    "exchange_exit_tier_filled",
                    {"order_id": order.id, "symbol": order.symbol, "tier": leg.tier,
                     "exit_price": order.exit_price, "filled_qty": executed,
                     "new_stop_price": plan.current_stop_price, "realized_pnl_so_far": order.realized_pnl},
                )
                changed = True
            elif status in _TERMINAL_STATUSES and executed <= 0:
                # Canceled/expired UNFILLED: drop pointers so it re-rests next tick.
                # A terminal order with partial fills must NOT rearm its full size
                # (that re-sells base already sold); mark it filled with what it got.
                leg.exit_order_exchange_id = None
                leg.exit_client_order_id = None
                changed = True
            elif status in _TERMINAL_STATUSES:
                apply_tier_fill(
                    plan, leg.tier, entry, self.exit_config, filled_qty=executed,
                    exit_order_exchange_id=leg.exit_order_exchange_id,
                    exit_client_order_id=leg.exit_client_order_id,
                )
                order.stop_loss_price = plan.current_stop_price
                self._compute_realized_pnl(order)
                changed = True
        held = order.executed_qty or order.quantity
        if remaining_quantity(plan, held) <= 1e-9 and all(leg.filled for leg in plan.legs):
            order.status = OrderStatus.CLOSED
            order.closed_at = utc_iso()
            self._compute_realized_pnl(order)
            self.store.log_event(
                "exchange_position_closed",
                {"order_id": order.id, "symbol": order.symbol, "exit_price": order.exit_price,
                 "exit_reason": order.exit_reason or "TAKE_PROFIT", "realized_pnl": order.realized_pnl,
                 "pnl_estimated": order.pnl_estimated, "balances_after_exit": self._balance_proof(order.symbol)},
            )
            changed = True
        if changed:
            self.store.save_order(order)
        return changed

    def _check_virtual_stop(self, order: OrderRecord, snapshot: MarketSnapshot) -> bool:
        """Trail the runner stop up; if price is at/under the (ratcheting) stop,
        cancel the resting take-profits and market-sell whatever remains."""
        plan = order.exit_plan
        if plan is None:
            return False
        mark = snapshot.last_price
        changed = update_trail(plan, mark, self.exit_config, atr=snapshot.atr)
        order.stop_loss_price = plan.current_stop_price
        already_exited = sum(f.qty for f in self.store.fills_for_order(order.id) if f.is_exit)
        remaining = (order.executed_qty or order.quantity) - already_exited
        if mark <= plan.current_stop_price and remaining > 1e-9:
            self._cancel_resting_legs(order)
            self._submit_exit(order, self._marketable_sell_price(order.symbol), stop_reason(plan), quantity=remaining)
            changed = True
        if changed:
            self.store.save_order(order)
        return changed

    def _cancel_resting_legs(self, order: OrderRecord) -> None:
        """Cancel any still-resting take-profit orders (before a stop-out/close so
        their locked balance is freed and we never orphan resting sells)."""
        plan = order.exit_plan
        if plan is None:
            return
        for leg in plan.legs:
            if leg.filled or leg.exit_order_exchange_id is None:
                continue
            try:
                self.adapter.cancel_order(
                    self.credentials, order.symbol,
                    order_id=leg.exit_order_exchange_id, client_order_id=leg.exit_client_order_id,
                )
            except RuntimeError as exc:
                if "-2011" not in str(exc):  # "already gone" is fine; otherwise warn
                    LOGGER.warning("cancel resting leg failed order_id=%s tier=%s: %s", order.id, leg.tier, exc)
            leg.exit_order_exchange_id = None
            leg.exit_client_order_id = None

    def _submit_exit(
        self, order: OrderRecord, price: float, reason: str, *, quantity: float | None = None
    ) -> None:
        """Submit the ORDER-LEVEL exit (stop-out / operator close); a market-ish
        sell of the remaining quantity. Tier take-profits use resting leg orders
        instead (_place_leg_order). The fill is confirmed in _sync_exit."""
        already_exited = sum(f.qty for f in self.store.fills_for_order(order.id) if f.is_exit)
        remaining = (order.executed_qty or order.quantity) - already_exited
        quantity_to_close = remaining if quantity is None else min(quantity, remaining)
        if quantity_to_close <= 0:
            order.status = OrderStatus.CLOSED
            order.closed_at = utc_iso()
            order.exit_reason = order.exit_reason or reason
            self._compute_realized_pnl(order)
            self.store.save_order(order)
            return
        quantity_q, quantized_price = self.adapter.quantize_order(order.symbol, quantity_to_close, price)
        client_order_id = f"tx{order.id.replace('_', '')[:20]}x"
        existing = self._query_order(order.symbol, order_id=None, client_order_id=client_order_id)
        if existing is not None and str(existing.get("status")) not in _TERMINAL_STATUSES:
            order.exit_order_exchange_id = str(existing.get("orderId"))
            order.exit_client_order_id = client_order_id
            order.exit_reason = order.exit_reason or reason
            order.closed_by = order.closed_by or reason
            self.store.log_event(
                "exchange_exit_adopted",
                {"order_id": order.id, "symbol": order.symbol,
                 "client_order_id": client_order_id, "exchange_order_id": order.exit_order_exchange_id},
            )
            return
        response = self.adapter.submit_limit_order(
            self.credentials, order.symbol, "SELL", quantity_q, quantized_price, client_order_id=client_order_id
        )
        raw = response.get("raw", {})
        balances_after_submit = self._balance_proof(order.symbol)
        order.exit_order_exchange_id = str(raw.get("orderId", "")) or None
        order.exit_client_order_id = client_order_id
        order.exit_reason = reason
        order.closed_by = reason
        LOGGER.info(
            "exit order submitted order_id=%s symbol=%s reason=%s price=%s qty=%s exchange_order_id=%s",
            order.id, order.symbol, reason, quantized_price, quantity_q, order.exit_order_exchange_id,
        )
        if balances_after_submit is not None:
            LOGGER.info("balances after exit submit order_id=%s symbol=%s balances=%s", order.id, order.symbol, balances_after_submit)
        self.store.log_event(
            "exchange_exit_submitted",
            {"order_id": order.id, "symbol": order.symbol, "reason": reason,
             "price": quantized_price, "quantity": quantity_q, "exchange_order_id": order.exit_order_exchange_id,
             "balances_after_submit": balances_after_submit},
        )

    @staticmethod
    def _bind_leg_exit(leg: ExitLeg | None, exchange_id: str | None, client_id: str) -> None:
        if leg is None:
            return
        leg.exit_order_exchange_id = exchange_id
        leg.exit_client_order_id = client_id

    def _sync_exit(self, order: OrderRecord) -> None:
        """Settle the ORDER-LEVEL exit (stop-out / operator close). On FILL the
        position fully closes; a terminal-incomplete exit clears the pointers so
        the next reconcile resubmits for the remainder. (Resting take-profit
        tiers are settled separately in _sync_resting_legs.)"""
        live = self._query_order(
            order.symbol,
            order_id=order.exit_order_exchange_id,
            client_order_id=order.exit_client_order_id,
        )
        if live is None:
            return
        exit_status = str(live.get("status", ""))
        executed = float(live.get("executedQty", 0) or 0)
        if executed > 0:
            self._ingest_fills(order, is_exit=True)
        if exit_status in _TERMINAL_STATUSES and exit_status != "FILLED":
            # The exit died (canceled/expired/rejected), possibly after partial
            # fills. Clear the exit pointers so the next reconcile resubmits for
            # the remaining quantity instead of wedging the position.
            self.store.log_event(
                "exchange_exit_terminal_incomplete",
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "exit_status": exit_status,
                    "exit_executed_qty": executed,
                },
            )
            order.exit_order_exchange_id = None
            order.exit_client_order_id = None
            return
        if exit_status != "FILLED":
            return
        executed_quote = float(live.get("cummulativeQuoteQty", 0) or 0)
        order.exit_price = round(executed_quote / executed, 8) if executed else order.exit_price
        order.status = OrderStatus.CLOSED
        order.closed_at = utc_iso()
        self._compute_realized_pnl(order)
        self.store.log_event(
            "exchange_position_closed",
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "exit_price": order.exit_price,
                "exit_reason": order.exit_reason,
                "realized_pnl": order.realized_pnl,
                "pnl_estimated": order.pnl_estimated,
                "quote_received": executed_quote,
                "balances_after_exit": self._balance_proof(order.symbol),
            },
        )

    def _compute_realized_pnl(self, order: OrderRecord) -> None:
        fills = self.store.fills_for_order(order.id)
        entry_fills = [f for f in fills if not f.is_exit]
        exit_fills = [f for f in fills if f.is_exit]
        if not entry_fills or not exit_fills:
            return
        base_asset, quote_asset = split_symbol(order.symbol)
        result = round_trip_pnl(
            entry_fills,
            exit_fills,
            base_asset=base_asset,
            quote_asset=quote_asset,
            conversion_prices=self._commission_conversion_prices(
                fills, base_asset=base_asset, quote_asset=quote_asset
            ),
        )
        order.realized_pnl = result.realized_pnl
        order.commission_total = result.commission_total_quote
        order.commission_asset = quote_asset
        order.pnl_estimated = result.estimated
        order.entry_fee = round(result.entry.commission_quote, 8)
        order.exit_fee = round(result.exit.commission_quote, 8)
        # Reflection journal (FinMem-style memory loop): on a FINAL close, record a
        # deterministic lesson so the next cycle's context carries what this trade
        # actually did. Guarded on CLOSED so partial tier fills (which also call
        # this) do not generate premature reflections. Idempotent per order_id.
        if order.status == OrderStatus.CLOSED:
            try:
                self.store.save_reflection(build_reflection(order))
            except Exception:  # memory must never break the close path
                LOGGER.exception("reflection save failed order_id=%s", order.id)

    def _commission_conversion_prices(
        self, fills: list[FillRecord], *, base_asset: str, quote_asset: str
    ) -> dict[str, float]:
        """Best-effort quote prices for commission assets outside base/quote
        (typically BNB with the fee discount on). Uses the current book price,
        so the PnL stays flagged estimated; but the fee is no longer ignored.
        """
        assets = {
            (fill.commission_asset or "").upper()
            for fill in fills
            if fill.commission and fill.commission_asset
        }
        assets -= {base_asset.upper(), quote_asset.upper(), ""}
        prices: dict[str, float] = {}
        for asset in sorted(assets):
            try:
                ticker = self.adapter.book_ticker(f"{asset}{quote_asset.upper()}")
                prices[asset] = float(ticker["bid_price"])
            except Exception as exc:
                LOGGER.warning("commission conversion price unavailable for %s: %s", asset, exc)
        return prices

    def _ingest_fills(
        self, order: OrderRecord, *, is_exit: bool, exchange_order_id: str | None = None
    ) -> int:
        # Explicit id lets us ingest a specific resting take-profit leg; otherwise
        # fall back to the order-level entry/exit ids.
        if exchange_order_id is None:
            exchange_order_id = order.exit_order_exchange_id if is_exit else order.exchange_order_id
        if exchange_order_id is None:
            return 0
        try:
            trades = self.adapter.get_my_trades(self.credentials, order.symbol, order_id=exchange_order_id)
        except RuntimeError as exc:
            LOGGER.warning("myTrades fetch failed order_id=%s: %s", order.id, exc)
            return 0
        fills = [
            FillRecord(
                order_id=order.id,
                exchange_order_id=str(trade.get("orderId")),
                exchange_trade_id=trade.get("id"),
                symbol=order.symbol,
                side=Side.SELL if is_exit else order.side,
                price=float(trade.get("price", 0)),
                qty=float(trade.get("qty", 0)),
                quote_qty=float(trade.get("quoteQty", 0)),
                commission=float(trade.get("commission", 0)),
                commission_asset=trade.get("commissionAsset"),
                is_exit=is_exit,
                trade_time=str(trade.get("time", "")),
                raw_json=None,
            )
            for trade in trades
        ]
        return self.store.save_fills(fills)

    def _query_order(
        self, symbol: str, *, order_id: str | None, client_order_id: str | None
    ) -> dict[str, Any] | None:
        """None means the venue no longer knows the order (e.g. testnet data wipe)."""
        try:
            return self.adapter.get_order(
                self.credentials, symbol, order_id=order_id, client_order_id=client_order_id
            )
        except RuntimeError as exc:
            if "-2013" in str(exc) or "does not exist" in str(exc).lower():
                return None
            raise

    def _cancel_on_exchange(self, order: OrderRecord) -> None:
        try:
            self.adapter.cancel_order(
                self.credentials,
                order.symbol,
                order_id=order.exchange_order_id,
                client_order_id=order.client_order_id,
            )
        except RuntimeError as exc:
            # Already gone (filled or expired) is fine; sync_order will catch up.
            if "-2011" not in str(exc):
                raise

    def _marketable_sell_price(self, symbol: str) -> float:
        ticker = self.adapter.book_ticker(symbol)
        return float(ticker["bid_price"])

    def _balance_proof(self, symbol: str) -> dict[str, Any] | None:
        """Base/quote free balances right now; the operator-facing evidence
        that a fill really moved funds (e.g. USDT down, SOL up). Best-effort:
        None when the account endpoint is unavailable."""
        try:
            balances = self.adapter.account_balances(self.credentials)
        except Exception as exc:
            LOGGER.warning("balance proof fetch failed symbol=%s: %s", symbol, exc)
            return None
        base_asset, quote_asset = split_symbol(symbol)
        by_asset = {entry["asset"]: entry for entry in balances}
        base = by_asset.get(base_asset, {})
        quote = by_asset.get(quote_asset, {})
        return {
            base_asset: {"free": base.get("free", 0.0), "locked": base.get("locked", 0.0)},
            quote_asset: {"free": quote.get("free", 0.0), "locked": quote.get("locked", 0.0)},
        }
