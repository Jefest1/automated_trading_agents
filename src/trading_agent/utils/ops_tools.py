"""Read-only operations tools for the /chat supervisor.

These let the agent answer questions about live orders and per-trade PnL.
They are strictly read-only: order placement/closing still flows exclusively
through the decision JSON -> risk gate -> execution pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from trading_agent.core.config import Settings
from trading_agent.core.exchange_sync import ExchangeReconciler
from trading_agent.core.logging import get_logger
from trading_agent.core.storage import Store
from trading_agent.exchange import BinanceSpotAdapter

LOGGER = get_logger("ops_tools")


def build_ops_tools(database_path: str | Path, settings: Settings) -> list[Any]:
    """Tools close over a Store factory (fresh connection per call: tool calls
    may arrive on any executor thread)."""

    def _open_store() -> Store:
        return Store(database_path)

    @tool
    def list_open_orders() -> str:
        """List currently open orders/positions with their order_id, entry price,
        TP/SL bracket, and last known exchange status. Use the order_id when
        proposing CLOSE or ADJUST decisions."""
        with _open_store() as store:
            orders = store.open_positions()
            payload = [
                {
                    "order_id": order.id,
                    "mode": order.mode,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "status": order.status.value,
                    "entry_price": order.price,
                    "quantity": order.quantity,
                    "executed_qty": order.executed_qty,
                    "avg_fill_price": order.avg_fill_price,
                    "take_profit_price": order.take_profit_price,
                    "stop_loss_price": order.stop_loss_price,
                    "exchange_status": order.exchange_status,
                    "opened_at": order.opened_at,
                }
                for order in orders
            ]
        return json.dumps({"ok": True, "open_orders": payload}, sort_keys=True)

    @tool
    def get_order_status(order_id: str) -> str:
        """Get one order's status. For testnet/live orders this queries the
        exchange LIVE (GET /api/v3/order) — never trust a stale local row."""
        with _open_store() as store:
            row = next((o for o in store.open_positions() if o.id == order_id), None)
            if row is None:
                match = [o for o in store.all_orders(200) if o.get("id") == order_id]
                if not match:
                    return json.dumps({"ok": False, "error": f"unknown order_id {order_id}"})
                return json.dumps({"ok": True, "source": "local_closed", "order": match[0]}, default=str)
        local = {
            "order_id": row.id,
            "mode": row.mode,
            "symbol": row.symbol,
            "status": row.status.value,
            "price": row.price,
            "quantity": row.quantity,
            "exchange_status": row.exchange_status,
        }
        if row.exchange_order_id is None:
            return json.dumps({"ok": True, "source": "local", "order": local}, sort_keys=True)
        try:
            credentials = BinanceSpotAdapter.credentials_from_env(settings=settings)
            adapter = BinanceSpotAdapter(base_url=settings.exchange_base_url(), settings=settings)
            live = adapter.get_order(credentials, row.symbol, order_id=row.exchange_order_id)
            local["live"] = {
                "status": live.get("status"),
                "executedQty": live.get("executedQty"),
                "cummulativeQuoteQty": live.get("cummulativeQuoteQty"),
                "price": live.get("price"),
            }
            return json.dumps({"ok": True, "source": "exchange", "order": local}, sort_keys=True)
        except Exception as exc:
            LOGGER.warning("live order status failed order_id=%s: %s", order_id, exc)
            local["live_error"] = str(exc)
            return json.dumps({"ok": True, "source": "local_stale", "order": local}, sort_keys=True)

    @tool
    def recent_trades_pnl(limit: int = 10) -> str:
        """Per-trade realized PnL for the most recent closed round trips
        (entry/exit fills and commissions included)."""
        with _open_store() as store:
            rows = store.per_trade_pnl(max(1, min(int(limit), 50)))
        return json.dumps({"ok": True, "closed_trades": rows}, sort_keys=True, default=str)

    @tool
    def recent_decisions(limit: int = 10) -> str:
        """Most recent supervisor decisions across cycles and operator chat:
        action, symbol, risk-gate approval with reasons, and the executed
        order id when one was placed. Use this to answer 'what did the agent
        decide (or propose) and why'."""
        with _open_store() as store:
            rows = store.recent_supervisor_decisions(max(1, min(int(limit), 50)))
        return json.dumps({"ok": True, "decisions": rows}, sort_keys=True, default=str)

    @tool
    def sync_orders_from_exchange() -> str:
        """Reconcile the LOCAL order table against the LIVE exchange: refresh
        status, fills, and PnL for open testnet/live orders and report what
        changed. Run this BEFORE acting on any open order so your view matches
        reality (a local row can be stale if a TP/SL already filled). Deterministic
        code performs the update; you only trigger it."""
        if settings.trading_agent_execution_mode not in {"testnet", "live"}:
            return json.dumps(
                {"ok": True, "updated": [], "note": "no exchange execution configured"}
            )
        with _open_store() as store:
            reconciler = ExchangeReconciler(store, settings)
            if not reconciler.available:
                return json.dumps({"ok": False, "error": "exchange credentials unavailable"})
            updated = reconciler.reconcile()
            changed = [
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "status": order.status.value,
                    "exchange_status": order.exchange_status,
                    "executed_qty": order.executed_qty,
                }
                for order in updated
            ]
        return json.dumps({"ok": True, "updated": changed, "count": len(changed)}, default=str)

    return [
        list_open_orders,
        get_order_status,
        recent_trades_pnl,
        recent_decisions,
        sync_orders_from_exchange,
    ]
