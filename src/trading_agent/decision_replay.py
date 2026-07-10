"""Frozen-context replay of recorded supervisor decisions.

The offline ``backtest`` measures the deterministic StrategyAgent. The thing
that actually trades in production is the LLM supervisor, and its decisions were
never scored against what happened next. This module closes that gap: it takes
the BUY decisions the supervisor actually recorded (with their limit price,
quantity, and bracket) and replays each through the same fill/TP/SL/fee model as
the backtester, against the real price path that followed the decision.

It does not re-invoke the model (its news/onchain context no longer exists); it
scores the decisions that were genuinely made. The decision journal in the DB is
the frozen context; this turns it into realized PnL, win rate, and a per-decision
buy-and-hold comparison so the LLM trader can finally be evaluated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from trading_agent.backtest import Candle, simulate_limit_entry
from trading_agent.core.config import AppConfig


@dataclass(slots=True)
class ReplayedDecision:
    decision_id: str
    symbol: str
    created_at: str
    executed: bool
    gate_approved: bool | None
    filled: bool
    entry_price: float
    quantity: float
    realized_pnl: float
    exit_reason: str | None
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "created_at": self.created_at,
            "executed": self.executed,
            "gate_approved": self.gate_approved,
            "filled": self.filled,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "realized_pnl_usd": round(self.realized_pnl, 8),
            "exit_reason": self.exit_reason,
            "note": self.note,
        }


@dataclass(slots=True)
class ReplayResult:
    decisions_total: int
    buy_decisions: int
    filled: int
    wins: int
    losses: int
    realized_pnl: float
    fees_paid: float
    buy_hold_pnl: float
    per_symbol_pnl: dict[str, float] = field(default_factory=dict)
    trades: list[ReplayedDecision] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "decisions_total": self.decisions_total,
            "buy_decisions": self.buy_decisions,
            "filled": self.filled,
            "unfilled": self.buy_decisions - self.filled,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / self.filled, 4) if self.filled else None,
            "realized_pnl_usd": round(self.realized_pnl, 8),
            "fees_paid_usd": round(self.fees_paid, 8),
            "buy_hold_pnl_usd": round(self.buy_hold_pnl, 8),
            "beats_buy_hold": self.realized_pnl > self.buy_hold_pnl,
            "per_symbol_pnl_usd": {k: round(v, 8) for k, v in sorted(self.per_symbol_pnl.items())},
            "trades": [trade.summary() for trade in self.trades],
        }


def candles_from_klines(rows: list[list[Any]]) -> list[Candle]:
    return [Candle.from_kline(row) for row in rows]


def replay_recorded_decisions(
    records: list[dict[str, Any]],
    windows: dict[str, list[Candle]],
    config: AppConfig,
    *,
    entry_ttl_candles: int = 3,
) -> ReplayResult:
    """Replay recorded supervisor decisions against per-decision forward windows.

    ``records`` are rows from ``Store.all_supervisor_decisions`` (payload parsed).
    ``windows`` maps a decision id to the forward candle window starting at/after
    that decision. WAIT/CLOSE/SELL/ADJUST decisions are counted but not traded.
    """
    fee_rate = config.backtest.fee_bps / 10_000
    slip_rate = config.backtest.slippage_bps / 10_000

    trades: list[ReplayedDecision] = []
    per_symbol: dict[str, float] = defaultdict(float)
    buy_decisions = filled = wins = losses = 0
    realized = fees = buy_hold = 0.0

    for rec in records:
        payload = rec.get("payload") or {}
        action = str(payload.get("action") or rec.get("action") or "").upper()
        if action != "BUY":
            continue
        buy_decisions += 1
        symbol = str(payload.get("symbol") or rec.get("symbol") or "")
        executed = bool(rec.get("executed_order_id"))
        gate_approved = rec.get("gate_approved")
        if gate_approved is not None:
            gate_approved = bool(gate_approved)
        created_at = str(rec.get("created_at") or payload.get("created_at") or "")
        limit_price = payload.get("limit_price")
        quantity = payload.get("quantity")
        candles = windows.get(str(rec.get("id")), [])

        if not candles or not limit_price or not quantity:
            trades.append(
                ReplayedDecision(
                    decision_id=str(rec.get("id")),
                    symbol=symbol,
                    created_at=created_at,
                    executed=executed,
                    gate_approved=gate_approved,
                    filled=False,
                    entry_price=float(limit_price or 0.0),
                    quantity=float(quantity or 0.0),
                    realized_pnl=0.0,
                    exit_reason=None,
                    note="no price window or incomplete decision",
                )
            )
            continue

        stop_loss_pct = payload.get("stop_loss_pct") or config.risk.stop_loss_pct
        take_profit_pct = payload.get("take_profit_pct") or config.risk.take_profit_pct
        notional = float(limit_price) * float(quantity)

        # Per-decision buy-and-hold over the same window and notional.
        first, last = candles[0].close, candles[-1].close
        hold_qty = notional / first if first else 0.0
        buy_hold += (last - first) * hold_qty - (notional * fee_rate + last * hold_qty * fee_rate)

        trade = simulate_limit_entry(
            symbol,
            candles,
            limit_price=float(limit_price),
            quantity=float(quantity),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
            fee_rate=fee_rate,
            slip_rate=slip_rate,
            entry_ttl_candles=entry_ttl_candles,
            exit_config=config.exits,
        )
        if trade is None:
            trades.append(
                ReplayedDecision(
                    decision_id=str(rec.get("id")),
                    symbol=symbol,
                    created_at=created_at,
                    executed=executed,
                    gate_approved=gate_approved,
                    filled=False,
                    entry_price=float(limit_price),
                    quantity=float(quantity),
                    realized_pnl=0.0,
                    exit_reason="UNFILLED",
                    note="limit never reached within entry TTL",
                )
            )
            continue

        filled += 1
        realized += trade.realized_pnl
        fees += trade.fees
        per_symbol[symbol] += trade.realized_pnl
        if trade.realized_pnl > 0:
            wins += 1
        else:
            losses += 1
        trades.append(
            ReplayedDecision(
                decision_id=str(rec.get("id")),
                symbol=symbol,
                created_at=created_at,
                executed=executed,
                gate_approved=gate_approved,
                filled=True,
                entry_price=trade.entry_price,
                quantity=trade.quantity,
                realized_pnl=trade.realized_pnl,
                exit_reason=trade.exit_reason,
            )
        )

    return ReplayResult(
        decisions_total=len(records),
        buy_decisions=buy_decisions,
        filled=filled,
        wins=wins,
        losses=losses,
        realized_pnl=realized,
        fees_paid=fees,
        buy_hold_pnl=buy_hold,
        per_symbol_pnl=dict(per_symbol),
        trades=trades,
    )
