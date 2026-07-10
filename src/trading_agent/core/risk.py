from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from trading_agent.core.config import AppConfig, RiskConfig, SizingConfig
from trading_agent.core.models import EvidenceRecord, MarketSnapshot, RiskDecision, TradeProposal
from trading_agent.core.storage import Store


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def conviction_size(
    *,
    confidence: float,
    expected_edge_bps: float,
    atr_pct: float | None,
    budget_usd: float,
    sizing: SizingConfig,
    regime_mult: float = 1.0,
    quality_mult: float = 1.0,
) -> float:
    """Target BUY notional (quote/USD), conviction-scaled.

    A just-valid setup takes ``min_size_frac`` of ``budget_usd``; strength
    (conviction x edge) ramps that toward ``kelly_fraction``. Volatility, regime,
    and data-quality multipliers only reduce the size. Returns 0.0 when the
    result is below the exchange minimum notional so a sub-minimum order is
    never sent. The RiskGovernor validates per-trade risk and exposure
    independently.
    """
    span = max(1e-9, sizing.conviction_full - sizing.conviction_floor)
    conviction_factor = _clamp((confidence - sizing.conviction_floor) / span, 0.0, 1.0)
    edge_factor = _clamp(expected_edge_bps / max(1e-9, sizing.edge_full_bps), 0.0, 1.0)
    # Both conviction and edge are required; either at zero yields floor size.
    strength = conviction_factor * edge_factor
    size_frac = sizing.min_size_frac + strength * (sizing.kelly_fraction - sizing.min_size_frac)
    size_frac = max(0.0, size_frac) * max(0.0, regime_mult) * max(0.0, quality_mult)
    notional = budget_usd * size_frac
    # Volatility targeting: trim size when current ATR% exceeds the target.
    if atr_pct and atr_pct > sizing.vol_target_pct:
        notional *= sizing.vol_target_pct / atr_pct
    ceiling = min(sizing.max_notional_usd, budget_usd)
    notional = min(notional, ceiling)
    if notional < sizing.min_notional_usd:
        return 0.0
    return round(notional, 8)


def maker_pullback_price(snapshot: MarketSnapshot, risk: RiskConfig) -> float:
    """Maker-pullback BUY limit: rests below the current bid so the order fills
    on a dip as a maker. Offset = entry_atr_mult * ATR, clamped to
    [entry_min_offset_bps, entry_max_offset_pct] of the bid. Shared by the
    StrategyAgent and the hard-maker gate override so both price identically."""
    bid = snapshot.bid_price
    atr_offset = risk.entry_atr_mult * (snapshot.atr or 0.0)
    min_offset = bid * risk.entry_min_offset_bps / 10_000
    max_offset = bid * risk.entry_max_offset_pct
    offset = max(min_offset, min(atr_offset, max_offset))
    return round(bid - offset, 8)


@dataclass(slots=True)
class RuntimeState:
    mode: str
    open_position_count: int
    kill_switch: bool
    open_notional_usd: float = 0.0
    available_quote_balance_usd: float | None = None
    # Open notional across the correlated majors (one beta bucket). For now all
    # tracked symbols are correlated majors, so this equals open_notional_usd.
    correlated_open_notional_usd: float = 0.0
    # Realized PnL for the current UTC day (drives the daily-loss circuit breaker).
    realized_pnl_today: float = 0.0


class RiskGovernor:
    def evaluate(
        self,
        proposal: TradeProposal,
        evidence: list[EvidenceRecord],
        state: RuntimeState,
        config: AppConfig,
    ) -> RiskDecision:
        reasons: list[str] = []
        risk = config.risk

        if state.kill_switch:
            reasons.append("kill switch is enabled")

        # Daily-loss circuit breaker: halt new entries once the day is down past
        # the configured fraction of the budget. Recorded as a breach, not a
        # routine veto (see RISK_BREACH_REASON_MARKERS).
        if risk.daily_loss_halt_pct > 0:
            daily_loss_limit = risk.daily_loss_halt_pct * config.live.capital_budget_usd
            if state.realized_pnl_today <= -daily_loss_limit:
                reasons.append(
                    f"daily loss limit reached ({state.realized_pnl_today:.2f} <= "
                    f"-{daily_loss_limit:.2f}); new entries halted"
                )

        if state.mode == "live":
            if not config.live.enabled:
                reasons.append("live trading is disabled")
            if not config.live.venue_confirmed:
                reasons.append("live venue/account constraints are not confirmed")
            if state.available_quote_balance_usd is None and proposal.notional_usd > config.live.capital_budget_usd:
                reasons.append("proposal notional exceeds live capital budget")

        if proposal.symbol not in risk.allowed_symbols:
            reasons.append(f"symbol {proposal.symbol} is not in the allowlist")

        if not evidence:
            reasons.append("proposal has no evidence")
        else:
            stale = [record.id for record in evidence if _age_seconds(record.observed_at) > risk.stale_data_seconds]
            if stale:
                reasons.append("proposal has stale evidence")
            placeholders = sorted({record.source for record in evidence if record.is_placeholder})
            if placeholders:
                reasons.append(
                    "proposal cites placeholder evidence (no live source): " + ", ".join(placeholders)
                )

        if state.open_position_count >= risk.max_open_positions:
            reasons.append("maximum open-position cap reached")

        if risk.max_open_positions <= 3 and state.open_position_count == risk.max_open_positions - 1:
            if proposal.confidence < risk.third_order_min_confidence:
                reasons.append("third open position requires at least 90% heuristic score")

        # Testnet uses a looser floor so paper trading still generates fills and
        # learning data; live keeps the stricter floor. Conviction-scaled sizing
        # makes the floor a noise gate, not an all-or-nothing participation gate.
        confidence_floor = risk.min_confidence
        if state.mode != "live":
            confidence_floor = min(risk.min_confidence, risk.testnet_min_confidence)
        if proposal.confidence < confidence_floor:
            reasons.append("proposal confidence is below minimum")

        if proposal.expected_edge_bps < risk.min_expected_edge_bps:
            reasons.append("expected edge is below minimum")

        # Worst-case loss is not stop_loss_pct alone: a limit stop can fill worse
        # than its price (slippage) and gap through it in a fast move, so both are
        # priced into the per-trade risk check.
        worst_case_loss_pct = (
            proposal.stop_loss_pct
            + risk.assumed_slippage_bps / 10_000
            + risk.stop_gap_buffer_pct
        )
        risk_amount = proposal.notional_usd * worst_case_loss_pct
        # Budget the trade against the live account: open notional + available quote
        # balance when known (the gate follows the real account, not a stale static
        # cap); otherwise the configured capital budget. Offline backtests pass a
        # large available balance so they gate on edge/confidence, not capital.
        if state.available_quote_balance_usd is not None:
            budget = state.open_notional_usd + state.available_quote_balance_usd
        else:
            budget = config.live.capital_budget_usd
        if risk_amount > budget * risk.per_trade_risk_fraction:
            reasons.append("per-trade risk exceeds configured risk fraction")

        if state.available_quote_balance_usd is not None and proposal.notional_usd > state.available_quote_balance_usd:
            reasons.append("proposal notional exceeds available quote balance")

        # Aggregate exposure: the position-count cap alone lets a few large
        # positions exceed the account/configured budget. In exchange-backed
        # modes the budget tracks available quote balance + already-open notional;
        # otherwise the configured capital budget, so the gate follows real funds.
        if state.open_notional_usd + proposal.notional_usd > budget:
            reasons.append("aggregate open notional would exceed available capital")

        # Correlated-exposure cap: BTC/ETH/SOL/BNB are one beta bet. Cap total open
        # notional across the correlated majors so a few "diversified" longs don't
        # quietly become one oversized leveraged-beta position. Healthy veto, not a
        # breach (the gate working).
        if risk.max_correlated_notional_usd > 0:
            projected = state.correlated_open_notional_usd + proposal.notional_usd
            if projected > risk.max_correlated_notional_usd:
                reasons.append(
                    f"aggregate correlated (major) exposure {projected:.2f} would exceed cap "
                    f"{risk.max_correlated_notional_usd:.2f}"
                )

        approved = not reasons
        return RiskDecision(proposal_id=proposal.id, approved=approved, reasons=reasons or ["approved"])

    @staticmethod
    def runtime_state(
        store: Store,
        mode: str,
        *,
        available_quote_balance_usd: float | None = None,
    ) -> RuntimeState:
        open_positions = store.open_positions()
        open_notional = 0.0
        for order in open_positions:
            # Executed quote spend when known, intended notional otherwise.
            open_notional += order.cumulative_quote_qty or (order.price * order.quantity)
        return RuntimeState(
            mode=mode,
            open_position_count=len(open_positions),
            kill_switch=bool(store.get_setting("kill_switch", False)),
            open_notional_usd=open_notional,
            available_quote_balance_usd=available_quote_balance_usd,
            # All tracked symbols are correlated majors -> one beta bucket.
            correlated_open_notional_usd=open_notional,
            realized_pnl_today=store.realized_pnl_today(),
        )


def _age_seconds(observed_at: str) -> float:
    parsed = datetime.fromisoformat(observed_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds()
