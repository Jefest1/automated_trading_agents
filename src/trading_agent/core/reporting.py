from __future__ import annotations

import json
from typing import Any

from trading_agent.core.config import AppConfig
from trading_agent.core.models import OrderStatus
from trading_agent.core.pnl import realized_from_fills, unrealized_pnl
from trading_agent.core.storage import Store


def promotion_report(
    store: Store,
    config: AppConfig,
    *,
    prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Operations report built around PER-TRADE PnL, not a single equity number.

    `prices` (symbol -> current price) is optional; when provided, open
    positions are marked to market so realized losses from closed trades are
    not mistaken for the whole story.
    """
    summary = store.summary()
    closed_trades = store.per_trade_pnl(20)
    realized_by_mode = store.realized_pnl_by_mode()
    realized_closed = round(sum(realized_by_mode.values()), 8)

    open_positions: list[dict[str, Any]] = []
    unrealized_total: float | None = 0.0
    # Interim realized PnL already BANKED by take-profit scale-outs on positions
    # that are still open (e.g. a filled TP1). It is real, locked-in cash, so it
    # counts toward realized_total alongside fully-closed round trips (shown as a
    # sub-line for transparency). Promotion is still gated separately on a minimum
    # number of CLOSED round trips, so folding banked cash in cannot promote early.
    realized_banked_open = 0.0
    for order in store.open_positions():
        current_price = (prices or {}).get(order.symbol)
        # Recompute banked PnL live from fills (not order.realized_pnl, which only
        # refreshes on reconcile and may be stale between cycles or after a fix).
        realized_banked_open += realized_from_fills(order, store.fills_for_order(order.id))
        entry = {
            "order_id": order.id,
            "mode": order.mode,
            "symbol": order.symbol,
            "status": order.status.value,
            "entry_price": order.price,
            "quantity": order.quantity,
            "executed_qty": order.executed_qty,
            "quote_spent": order.cumulative_quote_qty or None,
            "exchange_status": order.exchange_status,
            "take_profit_price": order.take_profit_price,
            "stop_loss_price": order.stop_loss_price,
            "current_price": current_price,
            "unrealized_pnl": None,
        }
        if order.status == OrderStatus.POSITION_OPEN and current_price is not None:
            # Marks only the HELD portion (subtracts filled TP tiers).
            unrealized, _ = unrealized_pnl(order, current_price)
            entry["unrealized_pnl"] = unrealized
            if unrealized is not None and unrealized_total is not None:
                unrealized_total += unrealized
        elif order.status == OrderStatus.POSITION_OPEN:
            unrealized_total = None  # at least one open position is unpriced
        open_positions.append(entry)

    # Realized total = fully-closed round trips + cash banked by scale-outs on
    # still-open positions (both are locked-in cash).
    realized_total = round(realized_closed + realized_banked_open, 8)

    # Net PnL must clear realized fees AND the LLM cost of running the desk —
    # a strategy whose edge is smaller than its own token bill is not promotable.
    llm_cost = store.get_setting("token_usage_cumulative", None) or {}
    llm_cost_usd = round(float(llm_cost.get("cost_usd", 0.0) or 0.0), 8)
    net_pnl_after_costs = round(realized_total - llm_cost_usd, 8)
    closed_count = len(store.per_trade_pnl(1000))
    min_trades = config.live.promotion_min_closed_trades

    positive_pnl = net_pnl_after_costs > 0
    # Real risk-control breaches only (kill switch, daily-loss halt, misconfig) —
    # NOT routine vetoes (confidence/edge/cap), which are the gate working.
    no_risk_breaches = summary["risk_breaches"] == 0
    enough_trades = closed_count >= min_trades
    return {
        "mode": config.mode,
        "promotion_gate": {
            "positive_net_pnl_after_costs": positive_pnl,
            "zero_risk_control_breaches": no_risk_breaches,
            "enough_closed_trades": enough_trades,
            "closed_trades": closed_count,
            "min_closed_trades_required": min_trades,
            "net_pnl_after_costs": net_pnl_after_costs,
            "llm_cost_usd": llm_cost_usd,
            "vetoes": summary.get("vetoes", 0),
            "eligible_for_live_capital_test": positive_pnl and no_risk_breaches and enough_trades,
        },
        "live_capital_test": {
            "enabled": config.live.enabled,
            "venue_confirmed": config.live.venue_confirmed,
            "budget_usd": config.live.capital_budget_usd,
            "budget_range_usd": [config.live.min_capital_budget_usd, config.live.max_capital_budget_usd],
        },
        "pnl": {
            "realized_total": realized_total,
            "realized_closed": realized_closed,
            "realized_banked_open": round(realized_banked_open, 8),
            "realized_by_mode": realized_by_mode,
            "unrealized_open_total": None if unrealized_total is None else round(unrealized_total, 8),
            "closed_trades": closed_trades,
        },
        "open_positions": open_positions,
        "summary": summary,
        "llm_cost": store.get_setting("token_usage_cumulative", None),
    }


def report_json(store: Store, config: AppConfig, *, prices: dict[str, float] | None = None) -> str:
    return json.dumps(promotion_report(store, config, prices=prices), indent=2, sort_keys=True, default=str)


def report_markdown(store: Store, config: AppConfig, *, prices: dict[str, float] | None = None) -> str:
    report = promotion_report(store, config, prices=prices)
    gate = report["promotion_gate"]
    summary = report["summary"]
    pnl = report["pnl"]
    lines = [
        "# Trading Agent Report",
        "",
        "## Promotion Gate",
        "",
        f"- Positive net PnL after costs (incl. LLM): {gate['positive_net_pnl_after_costs']} "
        f"(net {gate['net_pnl_after_costs']} = realized {pnl['realized_total']} - LLM {gate['llm_cost_usd']})",
        f"- Zero risk-control breaches: {gate['zero_risk_control_breaches']} "
        f"(routine vetoes, not blockers: {gate['vetoes']})",
        f"- Enough closed trades: {gate['enough_closed_trades']} "
        f"({gate['closed_trades']}/{gate['min_closed_trades_required']})",
        f"- Eligible for live-capital test: {gate['eligible_for_live_capital_test']}",
        "",
        "## PnL (per trade, after fees)",
        "",
        f"- Realized total (closed + banked): {pnl['realized_total']}",
        f"  - of which closed round trips: {pnl['realized_closed']}",
        f"  - of which banked from open scale-outs (TP tiers filled): {pnl['realized_banked_open']}",
        f"- Realized by mode (closed): `{pnl['realized_by_mode'] or 'none'}`",
        f"- Unrealized (open positions, held portion): "
        + (
            str(pnl["unrealized_open_total"])
            if pnl["unrealized_open_total"] is not None
            else "unknown (no live price for at least one open position)"
        ),
        "",
    ]
    if pnl["closed_trades"]:
        lines.append("### Closed trades")
        lines.append("")
        lines.append("| closed at | mode | symbol | entry | exit | reason | realized PnL | fees |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for trade in pnl["closed_trades"]:
            pnl_note = f"{trade['realized_pnl']}" + (" (est.)" if trade.get("pnl_estimated") else "")
            lines.append(
                f"| {str(trade.get('closed_at', ''))[:19]} | {trade['mode']} | {trade['symbol']} "
                f"| {trade['price']} | {trade.get('exit_price')} | {trade.get('exit_reason')} "
                f"| {pnl_note} | {trade.get('commission_total')} |"
            )
        lines.append("")
    if report["open_positions"]:
        lines.append("### Open positions")
        lines.append("")
        lines.append("| order id | mode | symbol | status | entry | current | unrealized | TP | SL |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for position in report["open_positions"]:
            lines.append(
                f"| {position['order_id']} | {position['mode']} | {position['symbol']} "
                f"| {position['status']} | {position['entry_price']} "
                f"| {position['current_price'] if position['current_price'] is not None else '-'} "
                f"| {position['unrealized_pnl'] if position['unrealized_pnl'] is not None else '-'} "
                f"| {position['take_profit_price']} | {position['stop_loss_price']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Runtime Summary",
            "",
            f"- Database: `{summary['database']}`",
            f"- Kill switch: {summary['kill_switch']}",
            f"- Evidence records: {summary['evidence_count']}",
            f"- Trade proposals: {summary['proposals_count']}",
            f"- Open positions: {summary['open_positions']}",
            # Vetoes are the gate WORKING (declining marginal trades) — expected.
            # Breaches are genuine control events (kill switch, daily-loss halt,
            # misconfig) — these are what the promotion gate cares about.
            f"- Gate vetoes (healthy, marginal trades declined): {summary.get('vetoes', 0)}",
            f"- Risk-control breaches (alarming): {summary['risk_breaches']}",
            f"- Order counts: `{summary['order_counts']}`",
            "",
        ]
    )
    return "\n".join(lines)
