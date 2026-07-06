"""Offline backtest: replay historical klines through the deterministic
pipeline (MarketDataAgent -> StrategyAgent -> RiskGovernor) with the backtest
fee/slippage model.

This measures the quant baseline against buy-and-hold and WAIT-always so
threshold changes (min edge, confidence, stops) can be evaluated on real
history instead of faith. The LLM supervisor is intentionally not replayed:
its decisions are non-deterministic and depend on live news/onchain context
that does not exist historically.

Candle mechanics:
- decisions are made on candle closes (no lookahead);
- a BUY limit at the decision candle's bid fills on the first later candle
  whose low touches the limit, and expires after ``entry_ttl_candles``;
- TP/SL are checked against each candle's high/low; if both are touched in
  the same candle the STOP is assumed first (conservative);
- fees apply on both sides, slippage on stop exits;
- anything still open at the end closes at the final close (END_OF_DATA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_agent.agents.signals import MarketDataAgent
from trading_agent.agents.strategy import StrategyAgent
from trading_agent.core.config import AppConfig
from trading_agent.core.logging import get_logger
from trading_agent.core.models import MarketSnapshot, TradeProposal, utc_iso
from trading_agent.core.risk import RiskGovernor, RuntimeState
from trading_agent.utils.indicators import atr, latest

LOGGER = get_logger("backtest")


@dataclass(slots=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: int

    @classmethod
    def from_kline(cls, row: list[Any]) -> Candle:
        return cls(
            open_time=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )


@dataclass(slots=True)
class BacktestTrade:
    symbol: str
    entry_index: int
    entry_price: float
    quantity: float
    exit_index: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    fees: float = 0.0
    realized_pnl: float = 0.0
    # Per-trade bracket (from the proposal); falls back to config when unset so the
    # legacy single-pct baseline is unchanged. Demand-zone bids carry their own
    # zone-derived stop/target here.
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass(slots=True)
class BacktestResult:
    symbol: str
    candles: int
    proposals: int
    risk_rejections: int
    trades: list[BacktestTrade] = field(default_factory=list)

    @property
    def realized_pnl(self) -> float:
        return round(sum(t.realized_pnl for t in self.trades), 8)

    @property
    def fees_paid(self) -> float:
        return round(sum(t.fees for t in self.trades), 8)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl <= 0)

    buy_hold_pnl: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "candles": self.candles,
            "proposals": self.proposals,
            "risk_rejections": self.risk_rejections,
            "trades": len(self.trades),
            "wins": self.wins,
            "losses": self.losses,
            "realized_pnl_usd": self.realized_pnl,
            "fees_paid_usd": self.fees_paid,
            "buy_hold_pnl_usd": round(self.buy_hold_pnl, 8),
            "wait_always_pnl_usd": 0.0,
            "beats_buy_hold": self.realized_pnl > self.buy_hold_pnl,
            "beats_wait_always": self.realized_pnl > 0.0,
        }


class _SeriesFeed:
    """MarketFeed over a candle series so MarketDataAgent runs unmodified."""

    def __init__(self, symbol: str, candles: list[Candle], spread_bps: float) -> None:
        self.symbol = symbol
        self.candles = candles
        self.spread_bps = spread_bps

    def snapshot(self, symbol: str, cycle: int) -> MarketSnapshot:
        index = max(0, min(cycle, len(self.candles) - 1))
        candle = self.candles[index]
        half_spread = candle.close * (self.spread_bps / 10_000) / 2
        window = self.candles[: index + 1]
        atr_value: float | None = None
        if len(window) >= 15:
            atr_value = latest(
                atr(
                    [c.high for c in window],
                    [c.low for c in window],
                    [c.close for c in window],
                )
            )
        return MarketSnapshot(
            symbol=symbol,
            observed_at=utc_iso(),
            last_price=candle.close,
            bid_price=candle.close - half_spread,
            ask_price=candle.close + half_spread,
            volume_24h=candle.volume,
            atr=atr_value,
        )


class Backtester:
    def __init__(
        self,
        config: AppConfig,
        *,
        spread_bps: float = 2.0,
        entry_ttl_candles: int = 3,
        zone_entry_ttl_candles: int = 12,
    ) -> None:
        self.config = config
        self.spread_bps = spread_bps
        self.entry_ttl_candles = entry_ttl_candles
        # Demand-zone bids rest deeper below price and need more candles to fill;
        # an un-anchored shallow maker entry keeps the tight legacy TTL.
        self.zone_entry_ttl_candles = zone_entry_ttl_candles
        self.strategy = StrategyAgent()
        self.risk = RiskGovernor()

    def run(self, symbol: str, klines: list[list[Any]]) -> BacktestResult:
        candles = [Candle.from_kline(row) for row in klines]
        result = BacktestResult(symbol=symbol, candles=len(candles), proposals=0, risk_rejections=0)
        if len(candles) < 3:
            return result

        feed = _SeriesFeed(symbol, candles, self.spread_bps)
        market_agent = MarketDataAgent(feed)
        fee_rate = self.config.backtest.fee_bps / 10_000
        slip_rate = self.config.backtest.slippage_bps / 10_000

        pending: tuple[TradeProposal, int, int] | None = None  # (proposal, decision index, ttl)
        open_trade: BacktestTrade | None = None

        for index in range(1, len(candles)):
            candle = candles[index]

            # 1) resolve the open position against this candle's range
            if open_trade is not None:
                trade = open_trade
                entry = trade.entry_price
                tp_pct = trade.take_profit_pct if trade.take_profit_pct is not None else self.config.risk.take_profit_pct
                sl_pct = trade.stop_loss_pct if trade.stop_loss_pct is not None else self.config.risk.stop_loss_pct
                tp = entry * (1 + tp_pct)
                sl = entry * (1 - sl_pct)
                exit_price: float | None = None
                reason: str | None = None
                if candle.low <= sl:
                    exit_price, reason = sl * (1 - slip_rate), "STOP_LOSS"
                elif candle.high >= tp:
                    exit_price, reason = tp, "TAKE_PROFIT"
                if exit_price is not None:
                    self._close(trade, index, exit_price, reason or "", fee_rate)
                    open_trade = None

            # 2) resolve a pending limit entry
            if pending is not None:
                proposal, decided_at, ttl = pending
                if candle.low <= proposal.price:
                    entry_fee = proposal.price * proposal.quantity * fee_rate
                    open_trade = BacktestTrade(
                        symbol=symbol,
                        entry_index=index,
                        entry_price=proposal.price,
                        quantity=proposal.quantity,
                        fees=round(entry_fee, 8),
                        stop_loss_pct=proposal.stop_loss_pct,
                        take_profit_pct=proposal.take_profit_pct,
                    )
                    result.trades.append(open_trade)
                    pending = None
                elif index - decided_at >= ttl:
                    pending = None  # unfilled entry expires

            # 3) decide on this candle's close
            if open_trade is None and pending is None:
                snapshot = feed.snapshot(symbol, index)
                evidence = market_agent.analyze(symbol, snapshot, index)
                # Build a demand-zone map from the candles seen so far (no lookahead)
                # so the deterministic baseline bids real support like the live desk.
                level_maps = self._level_maps(symbol, candles[: index + 1], snapshot.last_price)
                proposals = self.strategy.propose(
                    {symbol: snapshot}, [evidence], self.config, level_maps
                )
                if proposals:
                    result.proposals += 1
                    proposal = proposals[0]
                    # Offline historical sim: gate on edge/confidence/risk, not
                    # capital, so pass an effectively unbounded available balance.
                    state = RuntimeState(
                        mode="testnet",
                        open_position_count=0,
                        kill_switch=False,
                        open_notional_usd=0.0,
                        available_quote_balance_usd=1e12,
                    )
                    decision = self.risk.evaluate(proposal, [evidence], state, self.config)
                    if decision.approved:
                        ttl = self.zone_entry_ttl_candles if proposal.zone_id else self.entry_ttl_candles
                        pending = (proposal, index, ttl)
                    else:
                        result.risk_rejections += 1

        # 4) anything still open closes at the final close
        if open_trade is not None:
            self._close(open_trade, len(candles) - 1, candles[-1].close, "END_OF_DATA", fee_rate)

        notional = self.config.backtest.order_notional_usd
        first, last = candles[0].close, candles[-1].close
        hold_qty = notional / first
        hold_fees = notional * fee_rate + last * hold_qty * fee_rate
        result.buy_hold_pnl = (last - first) * hold_qty - hold_fees
        return result

    def _level_maps(
        self, symbol: str, window: list[Candle], price: float
    ) -> dict[str, Any]:
        """Demand-zone map from the candles seen so far (single backtest series).

        The backtest replays one timeframe; we feed it to the levels engine as the
        daily structure so swing pivots, volume nodes, fib, EMAs, and regime all
        engage. Returns {} (legacy maker entry) until there is enough history.
        """
        from trading_agent.core import levels as levels_engine

        if not self.config.risk.require_zone_anchored_bids or len(window) < 50:
            return {}
        lmap = levels_engine.build_level_map(symbol, {"1d": window}, price)
        return {symbol: lmap} if lmap.support_zones else {}

    @staticmethod
    def _close(
        trade: BacktestTrade, index: int, price: float, reason: str, fee_rate: float
    ) -> None:
        close_trade(trade, index, price, reason, fee_rate)


def close_trade(
    trade: BacktestTrade, index: int, price: float, reason: str, fee_rate: float
) -> None:
    """Settle a trade at ``price`` with exit fees and realized PnL."""
    exit_fee = price * trade.quantity * fee_rate
    trade.exit_index = index
    trade.exit_price = round(price, 8)
    trade.exit_reason = reason
    trade.fees = round(trade.fees + exit_fee, 8)
    gross = (price - trade.entry_price) * trade.quantity
    trade.realized_pnl = round(gross - trade.fees, 8)


def simulate_limit_entry(
    symbol: str,
    candles: list[Candle],
    *,
    limit_price: float,
    quantity: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    fee_rate: float,
    slip_rate: float,
    entry_ttl_candles: int = 3,
    exit_config: "ExitConfig | None" = None,
) -> BacktestTrade | None:
    """Replay one limit BUY with a TP/SL bracket over a forward candle window.

    candles[0] is the first candle at/after the decision. The limit fills on the
    first candle whose low touches ``limit_price`` within ``entry_ttl_candles``;
    after a fill, TP/SL are checked on *subsequent* candles (a fill and its exit
    never share a candle), with the stop assumed first if both are touched. A
    filled position still open at the end closes at the final close. Returns the
    settled trade, or None if the entry never filled. Mechanics mirror
    ``Backtester.run`` so replayed LLM decisions are scored apples-to-apples
    with the deterministic baseline.

    When ``exit_config`` is enabled the position is managed as a tiered scale-out
    ladder (TP1/TP2/runner with a ratcheting/trailing stop) instead of a single
    take-profit, so backtests reflect the live exit economics.
    """
    if exit_config is not None and exit_config.enabled:
        return _simulate_tiered_entry(
            symbol,
            candles,
            limit_price=limit_price,
            quantity=quantity,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            fee_rate=fee_rate,
            slip_rate=slip_rate,
            entry_ttl_candles=entry_ttl_candles,
            exit_config=exit_config,
        )
    trade: BacktestTrade | None = None
    for index, candle in enumerate(candles):
        if trade is None:
            if index >= entry_ttl_candles:
                return None  # entry expired unfilled
            if candle.low <= limit_price:
                entry_fee = limit_price * quantity * fee_rate
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_index=index,
                    entry_price=limit_price,
                    quantity=quantity,
                    fees=round(entry_fee, 8),
                )
            continue
        tp = trade.entry_price * (1 + take_profit_pct)
        sl = trade.entry_price * (1 - stop_loss_pct)
        if candle.low <= sl:
            close_trade(trade, index, sl * (1 - slip_rate), "STOP_LOSS", fee_rate)
            return trade
        if candle.high >= tp:
            close_trade(trade, index, tp, "TAKE_PROFIT", fee_rate)
            return trade
    if trade is not None:
        close_trade(trade, len(candles) - 1, candles[-1].close, "END_OF_DATA", fee_rate)
    return trade


def _simulate_tiered_entry(
    symbol: str,
    candles: list[Candle],
    *,
    limit_price: float,
    quantity: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    fee_rate: float,
    slip_rate: float,
    entry_ttl_candles: int,
    exit_config: "ExitConfig",
) -> BacktestTrade | None:
    """Tiered scale-out replay: partial take-profits on the way up + a stop that
    ratchets to breakeven, locks a prior tier, then trails the runner. Realized
    PnL accumulates across partials; ``exit_reason`` is the final exit's reason."""
    from trading_agent.core.exit_ladder import (
        apply_tier_fill,
        build_exit_plan,
        remaining_quantity,
        stop_reason,
    )

    plan = build_exit_plan(
        limit_price,
        exit_config,
        fallback_take_profit_pct=take_profit_pct,
        fallback_stop_loss_pct=stop_loss_pct,
    )
    trade: BacktestTrade | None = None

    def settle(price: float, qty: float) -> None:
        assert trade is not None
        exit_fee = price * qty * fee_rate
        gross = (price - trade.entry_price) * qty
        trade.fees = round(trade.fees + exit_fee, 8)
        trade.realized_pnl = round(trade.realized_pnl + gross - exit_fee, 8)
        trade.exit_price = round(price, 8)

    for index, candle in enumerate(candles):
        if trade is None:
            if index >= entry_ttl_candles:
                return None
            if candle.low <= limit_price:
                entry_fee = limit_price * quantity * fee_rate
                trade = BacktestTrade(
                    symbol=symbol,
                    entry_index=index,
                    entry_price=limit_price,
                    quantity=quantity,
                    fees=round(entry_fee, 8),
                    realized_pnl=round(-entry_fee, 8),  # pre-charge entry fee
                )
            continue
        # Stop first (conservative), against the pre-update stop on this candle's low.
        if candle.low <= plan.current_stop_price:
            settle(plan.current_stop_price * (1 - slip_rate), remaining_quantity(plan, quantity))
            trade.exit_index = index
            trade.exit_reason = stop_reason(plan)
            return trade
        # Then bank at most one take-profit tier touched by this candle's high.
        plan.high_water_price = max(plan.high_water_price, candle.high)
        for leg in plan.legs:
            if not leg.filled and candle.high >= leg.target_price:
                qty = quantity * leg.size_pct
                settle(leg.target_price, qty)
                apply_tier_fill(plan, leg.tier, limit_price, exit_config, filled_qty=qty)
                break
        # Trail the runner stop up from the high-water mark.
        if plan.runner_active and exit_config.trail_runner:
            candidate = plan.high_water_price * (1 - exit_config.trail_pct)
            if candidate > plan.current_stop_price:
                plan.current_stop_price = round(candidate, 8)
        if remaining_quantity(plan, quantity) <= 1e-12:
            trade.exit_index = index
            trade.exit_reason = "TAKE_PROFIT_ALL"
            return trade
    if trade is not None and remaining_quantity(plan, quantity) > 1e-12:
        settle(candles[-1].close, remaining_quantity(plan, quantity))
        trade.exit_index = len(candles) - 1
        trade.exit_reason = "END_OF_DATA"
    return trade
