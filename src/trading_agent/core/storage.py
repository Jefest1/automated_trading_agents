from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_agent.core.models import (
    EvidenceRecord,
    ExitLeg,
    ExitPlan,
    FillRecord,
    OrderRecord,
    OrderStatus,
    RiskDecision,
    Side,
    TradeIntent,
    TradeProposal,
    new_id,
    utc_iso,
)
from trading_agent.core.pnl import realized_from_fills


# Substrings in a risk-decision's reasons that mark a GENUINE risk-control event
# (alarming, should not happen in healthy operation), as opposed to the gate
# healthily declining a marginal or over-limit trade (a "veto"). Promotion
# requires zero breaches; vetoes are expected during selective trading. Kept here
# (not in risk.py) because risk.py already imports Store -> avoids a cycle.
RISK_BREACH_REASON_MARKERS = (
    "kill switch",
    "daily loss",
    "live trading is disabled",
    "venue/account constraints",
    "not in the allowlist",
    "has no evidence",
)


def is_risk_breach(reasons: list[str]) -> bool:
    text = " ".join(reasons).lower()
    return any(marker in text for marker in RISK_BREACH_REASON_MARKERS)


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    source TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_edge_bps REAL NOT NULL,
    risk_bps REAL NOT NULL,
    stop_loss_pct REAL NOT NULL,
    take_profit_pct REAL NOT NULL,
    rationale TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_decisions (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    approved INTEGER NOT NULL,
    reasons_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    entry_fee REAL NOT NULL,
    exit_fee REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    exit_price REAL,
    exit_reason TEXT,
    exchange_order_id TEXT,
    client_order_id TEXT,
    exchange_status TEXT,
    executed_qty REAL NOT NULL DEFAULT 0,
    cumulative_quote_qty REAL NOT NULL DEFAULT 0,
    avg_fill_price REAL,
    commission_total REAL NOT NULL DEFAULT 0,
    commission_asset TEXT,
    pnl_estimated INTEGER NOT NULL DEFAULT 0,
    last_synced_at TEXT,
    decision_id TEXT,
    closed_by TEXT,
    exit_order_exchange_id TEXT,
    exit_client_order_id TEXT,
    exit_plan_json TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    exchange_order_id TEXT,
    exchange_trade_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    quote_qty REAL NOT NULL,
    commission REAL NOT NULL,
    commission_asset TEXT,
    is_exit INTEGER NOT NULL DEFAULT 0,
    trade_time TEXT,
    raw_json TEXT,
    UNIQUE(symbol, exchange_trade_id)
);

CREATE TABLE IF NOT EXISTS supervisor_decisions (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    symbol TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    gate_approved INTEGER,
    gate_reasons_json TEXT,
    executed_order_id TEXT,
    source TEXT NOT NULL DEFAULT 'supervisor'
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    cycle INTEGER NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS reflections (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    order_id TEXT UNIQUE,
    symbol TEXT NOT NULL,
    outcome TEXT NOT NULL,
    realized_pnl REAL NOT NULL,
    realized_r REAL,
    exit_reason TEXT,
    holding_minutes REAL,
    lesson TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    agent_name TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_summary TEXT NOT NULL,
    output_summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_call_logs (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input_summary TEXT NOT NULL,
    output_summary TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_intents (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    created_at TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    limit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_edge_bps REAL NOT NULL,
    risk_bps REAL,
    stop_loss_pct REAL NOT NULL,
    take_profit_pct REAL NOT NULL,
    rationale TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: under graph.ainvoke, LangGraph runs sync
        # nodes on executor threads. Access stays serialized by the runtime's
        # invocation lock; SQLite itself is compiled threadsafe.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def log_event(self, category: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events (created_at, category, payload_json) VALUES (?, ?, ?)",
            (utc_iso(), category, json.dumps(payload, sort_keys=True)),
        )
        self.conn.commit()

    def save_evidence(self, records: Iterable[EvidenceRecord]) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO evidence
            (id, agent, source, symbol, kind, observed_at, score, confidence, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.id,
                    r.agent,
                    r.source,
                    r.symbol,
                    r.kind,
                    r.observed_at,
                    r.score,
                    r.confidence,
                    json.dumps(r.payload, sort_keys=True),
                )
                for r in records
            ],
        )
        self.conn.commit()

    def save_proposal(self, proposal: TradeProposal) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO proposals
            (id, created_at, symbol, side, price, quantity, confidence, expected_edge_bps,
             risk_bps, stop_loss_pct, take_profit_pct, rationale, evidence_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal.id,
                proposal.created_at,
                proposal.symbol,
                proposal.side.value,
                proposal.price,
                proposal.quantity,
                proposal.confidence,
                proposal.expected_edge_bps,
                proposal.risk_bps,
                proposal.stop_loss_pct,
                proposal.take_profit_pct,
                proposal.rationale,
                json.dumps(proposal.evidence_ids),
            ),
        )
        self.conn.commit()

    def save_risk_decision(self, decision: RiskDecision) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO risk_decisions
            (id, proposal_id, approved, reasons_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision.id,
                decision.proposal_id,
                int(decision.approved),
                json.dumps(decision.reasons),
                decision.created_at,
            ),
        )
        self.conn.commit()

    def save_trade_intent(self, intent: TradeIntent, run_id: str | None = None) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trade_intents
            (id, run_id, created_at, source_agent, symbol, side, limit_price, quantity,
             confidence, expected_edge_bps, risk_bps, stop_loss_pct, take_profit_pct,
             rationale, evidence_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent.id,
                run_id,
                intent.created_at,
                intent.source_agent,
                intent.symbol,
                intent.side.value,
                intent.limit_price,
                intent.quantity,
                intent.confidence,
                intent.expected_edge_bps,
                intent.risk_bps,
                intent.stop_loss_pct,
                intent.take_profit_pct,
                intent.rationale,
                json.dumps(intent.evidence_ids),
            ),
        )
        self.conn.commit()

    def save_order(self, order: OrderRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO orders
            (id, proposal_id, mode, symbol, side, order_type, price, quantity,
             take_profit_price, stop_loss_price, status, opened_at, closed_at,
             entry_fee, exit_fee, realized_pnl, exit_price, exit_reason,
             exchange_order_id, client_order_id, exchange_status, executed_qty,
             cumulative_quote_qty, avg_fill_price, commission_total, commission_asset,
             pnl_estimated, last_synced_at, decision_id, closed_by,
             exit_order_exchange_id, exit_client_order_id, exit_plan_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.id,
                order.proposal_id,
                order.mode,
                order.symbol,
                order.side.value,
                order.order_type,
                order.price,
                order.quantity,
                order.take_profit_price,
                order.stop_loss_price,
                order.status.value,
                order.opened_at,
                order.closed_at,
                order.entry_fee,
                order.exit_fee,
                order.realized_pnl,
                order.exit_price,
                order.exit_reason,
                order.exchange_order_id,
                order.client_order_id,
                order.exchange_status,
                order.executed_qty,
                order.cumulative_quote_qty,
                order.avg_fill_price,
                order.commission_total,
                order.commission_asset,
                int(order.pnl_estimated),
                order.last_synced_at,
                order.decision_id,
                order.closed_by,
                order.exit_order_exchange_id,
                order.exit_client_order_id,
                order.exit_plan.to_json() if order.exit_plan is not None else None,
            ),
        )
        self.conn.commit()

    def save_fills(self, fills: Iterable[FillRecord]) -> int:
        """Idempotent fill upsert: duplicates on (symbol, exchange_trade_id) are ignored."""
        before = self.conn.total_changes
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO fills
            (id, order_id, exchange_order_id, exchange_trade_id, symbol, side,
             price, qty, quote_qty, commission, commission_asset, is_exit,
             trade_time, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f.id,
                    f.order_id,
                    f.exchange_order_id,
                    f.exchange_trade_id,
                    f.symbol,
                    f.side.value,
                    f.price,
                    f.qty,
                    f.quote_qty,
                    f.commission,
                    f.commission_asset,
                    int(f.is_exit),
                    f.trade_time,
                    f.raw_json,
                )
                for f in fills
            ],
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def fills_for_order(self, order_id: str) -> list[FillRecord]:
        rows = self.conn.execute(
            "SELECT * FROM fills WHERE order_id = ? ORDER BY trade_time", (order_id,)
        ).fetchall()
        return [
            FillRecord(
                id=row["id"],
                order_id=row["order_id"],
                exchange_order_id=row["exchange_order_id"],
                exchange_trade_id=row["exchange_trade_id"],
                symbol=row["symbol"],
                side=Side(row["side"]),
                price=float(row["price"]),
                qty=float(row["qty"]),
                quote_qty=float(row["quote_qty"]),
                commission=float(row["commission"]),
                commission_asset=row["commission_asset"],
                is_exit=bool(row["is_exit"]),
                trade_time=row["trade_time"],
                raw_json=row["raw_json"],
            )
            for row in rows
        ]

    def save_supervisor_decision(
        self,
        *,
        decision_id: str,
        run_id: str | None,
        action: str,
        symbol: str,
        payload: dict[str, Any],
        gate_approved: bool | None = None,
        gate_reasons: list[str] | None = None,
        executed_order_id: str | None = None,
        source: str = "supervisor",
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO supervisor_decisions
            (id, run_id, created_at, action, symbol, payload_json,
             gate_approved, gate_reasons_json, executed_order_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                run_id,
                utc_iso(),
                action,
                symbol,
                json.dumps(payload, sort_keys=True, default=str),
                None if gate_approved is None else int(gate_approved),
                json.dumps(gate_reasons or []),
                executed_order_id,
                source,
            ),
        )
        self.conn.commit()

    def recent_supervisor_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM supervisor_decisions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            self._row_dict(row, json_fields=("payload_json", "gate_reasons_json"))
            for row in rows
        ]

    def all_supervisor_decisions(self, limit: int = 100_000) -> list[dict[str, Any]]:
        """Every recorded supervisor decision, oldest first (replay chronology)."""
        rows = self.conn.execute(
            "SELECT * FROM supervisor_decisions ORDER BY created_at ASC LIMIT ?", (limit,)
        ).fetchall()
        return [
            self._row_dict(row, json_fields=("payload_json", "gate_reasons_json"))
            for row in rows
        ]

    def per_trade_pnl(self, limit: int = 50) -> list[dict[str, Any]]:
        """Realized PnL per closed round trip (one row per order), newest first."""
        rows = self.conn.execute(
            """
            SELECT id, mode, symbol, side, price, quantity, executed_qty, avg_fill_price,
                   exit_price, exit_reason, realized_pnl, commission_total, commission_asset,
                   pnl_estimated, opened_at, closed_at, closed_by
            FROM orders
            WHERE status = 'CLOSED'
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def open_orders(self) -> list[OrderRecord]:
        return self.open_positions()

    def open_positions(self) -> list[OrderRecord]:
        # PENDING_SUBMIT counts as open: the order may already exist on the
        # venue, so position caps and the reconciler must see it.
        rows = self.conn.execute(
            "SELECT * FROM orders WHERE status IN ('PENDING_SUBMIT', 'ENTRY_OPEN', 'POSITION_OPEN')"
            " ORDER BY opened_at"
        ).fetchall()
        return [self._row_to_order(row) for row in rows]

    def recent_evidence(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM evidence ORDER BY observed_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_dict(row, json_fields=("payload_json",)) for row in rows]

    def all_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM orders ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def count_vetoes(self) -> int:
        """Intents the deterministic gate DECLINED. A veto is the gate WORKING
        (a marginal/over-limit trade refused); it is expected during normal
        selective trading and must NOT block promotion."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM risk_decisions WHERE approved = 0"
        ).fetchone()
        return int(row["count"])

    def count_risk_breaches(self) -> int:
        """Genuine risk-control events (alarming), NOT routine vetoes.

        Previously this counted every rejected intent, so normal confidence/edge
        vetoes (e.g. conf 0.64 < 0.65) showed up as "breaches" and made the
        zero-breach promotion gate unsatisfiable. Now only rejections whose
        reasons signal a real control event (kill switch, daily-loss halt, live
        misconfiguration, allowlist/no-evidence bugs) count."""
        rows = self.conn.execute(
            "SELECT reasons_json FROM risk_decisions WHERE approved = 0"
        ).fetchall()
        return sum(1 for row in rows if is_risk_breach(json.loads(row["reasons_json"] or "[]")))

    def count_trade_intents(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS count FROM trade_intents").fetchone()
        return int(row["count"])

    def count_prompt_logs(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS count FROM prompt_logs").fetchone()
        return int(row["count"])

    def banked_realized_open(self) -> float:
        """Interim realized PnL banked by scale-outs on still-OPEN positions,
        recomputed live FROM FILLS (not the cached order.realized_pnl field, which
        only refreshes on reconcile and can lag/stale). 0.0 when nothing exited."""
        total = sum(
            realized_from_fills(order, self.fills_for_order(order.id))
            for order in self.open_positions()
        )
        return round(total, 8)

    def realized_pnl(self) -> float:
        """Total realized cash: fully-closed round trips (their canonical stored
        value, computed at close) PLUS interim banked PnL on open positions,
        recomputed from fills so a stale cached value never surfaces."""
        closed = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM orders WHERE status = 'CLOSED'"
        ).fetchone()
        return round(float(closed["pnl"]) + self.banked_realized_open(), 8)

    def save_reflection(self, reflection: Any) -> None:
        """Persist a post-trade reflection. Idempotent per order_id (UNIQUE), so
        re-running the close path never duplicates a lesson."""
        self.conn.execute(
            """
            INSERT OR IGNORE INTO reflections
            (id, created_at, order_id, symbol, outcome, realized_pnl, realized_r,
             exit_reason, holding_minutes, lesson)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reflection.id,
                reflection.created_at,
                reflection.order_id,
                reflection.symbol,
                reflection.outcome,
                float(reflection.realized_pnl),
                reflection.realized_r,
                reflection.exit_reason,
                reflection.holding_minutes,
                reflection.lesson,
            ),
        )
        self.conn.commit()

    def recent_reflections(self, limit: int = 10, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol is not None:
            rows = self.conn.execute(
                "SELECT * FROM reflections WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def trade_stats(self) -> dict[str, Any]:
        """Aggregate realized-trade calibration: win rate, average R, expectancy.

        This is the feedback signal injected into the cycle context so the desk's
        conviction is grounded in what its recent trades actually returned."""
        pnls = [
            float(row["realized_pnl"])
            for row in self.conn.execute(
                "SELECT realized_pnl FROM orders WHERE status = 'CLOSED'"
            ).fetchall()
        ]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        total = round(sum(pnls), 8)
        avg_r_row = self.conn.execute(
            "SELECT AVG(realized_r) AS avg_r FROM reflections WHERE realized_r IS NOT NULL"
        ).fetchone()
        avg_r = avg_r_row["avg_r"] if avg_r_row else None
        return {
            "closed_trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / n, 4) if n else None,
            "avg_pnl": round(total / n, 8) if n else None,
            "avg_realized_r": round(float(avg_r), 3) if avg_r is not None else None,
            "total_realized_pnl": total,
        }

    def realized_pnl_today(self) -> float:
        """Realized PnL from round trips closed during the current UTC day.

        Drives the daily-loss circuit breaker. ISO-8601 closed_at strings sort
        lexicographically, so a date-prefix comparison isolates today's closes."""
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM orders "
            "WHERE status = 'CLOSED' AND closed_at >= ?",
            (day,),
        ).fetchone()
        return float(row["pnl"])

    def realized_pnl_by_mode(self) -> dict[str, float]:
        rows = self.conn.execute(
            """
            SELECT mode, COALESCE(SUM(realized_pnl), 0) AS pnl, COUNT(*) AS trades
            FROM orders WHERE status = 'CLOSED' GROUP BY mode
            """
        ).fetchall()
        return {row["mode"]: round(float(row["pnl"]), 8) for row in rows}

    def order_counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS count FROM orders GROUP BY status").fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def set_setting(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO settings (key, value_json) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value_json FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    def start_agent_run(self, thread_id: str, cycle: int) -> str:
        run_id = new_id("run")
        self.conn.execute(
            """
            INSERT INTO agent_runs (id, thread_id, status, started_at, cycle)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, thread_id, "running", utc_iso(), cycle),
        )
        self.conn.commit()
        return run_id

    def finish_agent_run(self, run_id: str, status: str, error: str | None = None) -> None:
        self.conn.execute(
            """
            UPDATE agent_runs
            SET status = ?, finished_at = ?, error = ?
            WHERE id = ?
            """,
            (status, utc_iso(), error, run_id),
        )
        self.conn.commit()

    def save_prompt_log(
        self,
        *,
        run_id: str | None,
        agent_name: str,
        prompt_name: str,
        prompt_version: str,
        input_summary: str,
        output_summary: str,
    ) -> str:
        log_id = new_id("plog")
        self.conn.execute(
            """
            INSERT INTO prompt_logs
            (id, run_id, agent_name, prompt_name, prompt_version, input_summary, output_summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                run_id,
                agent_name,
                prompt_name,
                prompt_version,
                input_summary,
                output_summary,
                utc_iso(),
            ),
        )
        self.conn.commit()
        return log_id

    def save_tool_call_log(
        self,
        *,
        run_id: str | None,
        agent_name: str,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        error: str | None = None,
    ) -> str:
        log_id = new_id("tlog")
        self.conn.execute(
            """
            INSERT INTO tool_call_logs
            (id, run_id, agent_name, tool_name, input_summary, output_summary, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                run_id,
                agent_name,
                tool_name,
                input_summary,
                output_summary,
                error,
                utc_iso(),
            ),
        )
        self.conn.commit()
        return log_id

    def latest_agent_run(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(row)

    def set_heartbeat(self, payload: dict[str, Any]) -> None:
        data = dict(payload)
        data["updated_at"] = utc_iso()
        self.set_setting("agent_heartbeat", data)

    def heartbeat(self) -> dict[str, Any] | None:
        return self.get_setting("agent_heartbeat", None)

    def try_acquire_agent_lock(self, owner: str, stale_after_seconds: int = 3600) -> bool:
        current = self.get_setting("agent_lock", None)
        if current:
            acquired_at = _parse_iso(current.get("acquired_at"))
            if acquired_at is not None:
                age = (datetime.now(UTC) - acquired_at).total_seconds()
                if age < stale_after_seconds:
                    return False
        self.set_setting("agent_lock", {"owner": owner, "acquired_at": utc_iso()})
        return True

    def release_agent_lock(self, owner: str) -> None:
        current = self.get_setting("agent_lock", None)
        if current and current.get("owner") != owner:
            return
        self.conn.execute("DELETE FROM settings WHERE key = ?", ("agent_lock",))
        self.conn.commit()

    def summary(self) -> dict[str, Any]:
        evidence_count = self.conn.execute("SELECT COUNT(*) AS count FROM evidence").fetchone()["count"]
        proposals_count = self.conn.execute("SELECT COUNT(*) AS count FROM proposals").fetchone()["count"]
        return {
            "database": str(self.path),
            "kill_switch": bool(self.get_setting("kill_switch", False)),
            "evidence_count": int(evidence_count),
            "proposals_count": int(proposals_count),
            "trade_intents_count": self.count_trade_intents(),
            "prompt_logs_count": self.count_prompt_logs(),
            "order_counts": self.order_counts(),
            "open_positions": len(self.open_positions()),
            "realized_pnl": round(self.realized_pnl(), 8),
            "risk_breaches": self.count_risk_breaches(),
            "vetoes": self.count_vetoes(),
            "latest_agent_run": self.latest_agent_run(),
            "heartbeat": self.heartbeat(),
            "agent_checkpoint": self.get_setting("agent_checkpoint", None),
        }

    _ORDER_COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("exit_reason", "TEXT"),
        ("exchange_order_id", "TEXT"),
        ("client_order_id", "TEXT"),
        ("exchange_status", "TEXT"),
        ("executed_qty", "REAL NOT NULL DEFAULT 0"),
        ("cumulative_quote_qty", "REAL NOT NULL DEFAULT 0"),
        ("avg_fill_price", "REAL"),
        ("commission_total", "REAL NOT NULL DEFAULT 0"),
        ("commission_asset", "TEXT"),
        ("pnl_estimated", "INTEGER NOT NULL DEFAULT 0"),
        ("last_synced_at", "TEXT"),
        ("decision_id", "TEXT"),
        ("closed_by", "TEXT"),
        ("exit_order_exchange_id", "TEXT"),
        ("exit_client_order_id", "TEXT"),
        ("exit_plan_json", "TEXT"),
    )

    def _migrate(self) -> None:
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(orders)").fetchall()
        }
        for name, ddl in self._ORDER_COLUMN_MIGRATIONS:
            if name not in columns:
                self.conn.execute(f"ALTER TABLE orders ADD COLUMN {name} {ddl}")
        self.conn.execute(
            "UPDATE orders SET status = 'POSITION_OPEN' WHERE status = 'EXIT_OPEN'"
        )
        self.conn.execute(
            "UPDATE orders SET order_type = 'SPOT_LIMIT_ENTRY' WHERE order_type = 'LIMIT_WITH_OCO_EXIT'"
        )

    def _row_to_order(self, row: sqlite3.Row) -> OrderRecord:
        return OrderRecord(
            id=row["id"],
            proposal_id=row["proposal_id"],
            mode=row["mode"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            order_type=row["order_type"],
            price=float(row["price"]),
            quantity=float(row["quantity"]),
            take_profit_price=float(row["take_profit_price"]),
            stop_loss_price=float(row["stop_loss_price"]),
            status=OrderStatus(row["status"]),
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            entry_fee=float(row["entry_fee"]),
            exit_fee=float(row["exit_fee"]),
            realized_pnl=float(row["realized_pnl"]),
            exit_price=row["exit_price"],
            exit_reason=row["exit_reason"],
            exchange_order_id=row["exchange_order_id"],
            client_order_id=row["client_order_id"],
            exchange_status=row["exchange_status"],
            executed_qty=float(row["executed_qty"] or 0),
            cumulative_quote_qty=float(row["cumulative_quote_qty"] or 0),
            avg_fill_price=row["avg_fill_price"],
            commission_total=float(row["commission_total"] or 0),
            commission_asset=row["commission_asset"],
            pnl_estimated=bool(row["pnl_estimated"]),
            last_synced_at=row["last_synced_at"],
            decision_id=row["decision_id"],
            closed_by=row["closed_by"],
            exit_order_exchange_id=row["exit_order_exchange_id"],
            exit_client_order_id=row["exit_client_order_id"],
            exit_plan=self._load_exit_plan(row),
        )

    @staticmethod
    def _load_exit_plan(row: sqlite3.Row) -> ExitPlan | None:
        """Deserialize the ladder, or synthesize a legacy single-leg plan from
        the stored tp/sl so rows written before tiered exits still manage."""
        keys = row.keys()
        raw = row["exit_plan_json"] if "exit_plan_json" in keys else None
        plan = ExitPlan.from_json(raw)
        if plan is not None:
            return plan
        entry = float(row["price"])
        return ExitPlan(
            legs=[ExitLeg(tier=1, target_price=float(row["take_profit_price"]), size_pct=1.0)],
            initial_stop_price=float(row["stop_loss_price"]),
            current_stop_price=float(row["stop_loss_price"]),
            high_water_price=entry,
            runner_size_pct=0.0,
            tiered=False,
        )

    @staticmethod
    def _row_dict(row: sqlite3.Row, json_fields: tuple[str, ...] = ()) -> dict[str, Any]:
        data = dict(row)
        for field in json_fields:
            if field in data:
                data[field.replace("_json", "")] = json.loads(data.pop(field))
        return data


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
