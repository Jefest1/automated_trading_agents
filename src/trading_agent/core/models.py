from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso() -> str:
    return utc_now().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    # Row persisted before the exchange call so a crash mid-submit leaves a
    # client_order_id the reconciler can resolve against the venue.
    PENDING_SUBMIT = "PENDING_SUBMIT"
    ENTRY_OPEN = "ENTRY_OPEN"
    POSITION_OPEN = "POSITION_OPEN"
    CLOSED = "CLOSED"
    CANCELED = "CANCELED"


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    observed_at: str
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    # ATR(15m) when available, used to size maker-pullback entries below the bid.
    atr: float | None = None


# Evidence sources ending in this suffix are deterministic stand-ins emitted
# when no live data source was reachable. They keep the pipeline observable
# but must never contribute to trade decisions.
PLACEHOLDER_SOURCE_SUFFIX = "-placeholder"

# Honest down-weighting: not all live sources are equally trustworthy. A primary
# source (the exchange feed, the Binance skills hub) earns full weight; a degraded
# fallback (broad web news, a coarse TVL/flow proxy) is real but noisier, so it is
# weighted down instead of trusted equally or silently treated as a clean signal.
# Quality in (0, 1]; placeholders are excluded entirely (quality 0).
_SOURCE_QUALITY: dict[str, float] = {
    "binance-compatible-feed": 1.0,
    "binance-skills-hub": 1.0,
    "web-news": 0.7,
    "gdelt-news": 0.6,
    "defillama-tvl": 0.65,
}
_DEFAULT_SOURCE_QUALITY = 0.75


@dataclass(slots=True)
class EvidenceRecord:
    agent: str
    source: str
    symbol: str
    kind: str
    observed_at: str
    score: float
    confidence: float
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: new_id("ev"))

    @property
    def is_placeholder(self) -> bool:
        return self.source.endswith(PLACEHOLDER_SOURCE_SUFFIX)

    @property
    def quality(self) -> float:
        """Source-quality weight in (0, 1]; 0 for placeholders. Used to
        down-weight degraded fallback sources in scoring and sizing."""
        if self.is_placeholder:
            return 0.0
        return _SOURCE_QUALITY.get(self.source, _DEFAULT_SOURCE_QUALITY)

    @property
    def is_degraded(self) -> bool:
        """A real but lower-quality fallback source (not primary, not placeholder)."""
        return not self.is_placeholder and self.quality < 1.0


@dataclass(slots=True)
class TradeProposal:
    symbol: str
    side: Side
    price: float
    quantity: float
    confidence: float
    expected_edge_bps: float
    risk_bps: float
    stop_loss_pct: float
    take_profit_pct: float
    rationale: str
    evidence_ids: list[str]
    # Demand-zone laddered entry context (None for legacy/shallow maker entries):
    # absolute target/stop derived from the analyst-confirmed zone, the zone id this
    # bid rests in, a shared ladder id grouping the scaled legs, and an expiry stamp
    # (GTD-style) after which an unfilled bid is cancelled.
    target_price: float | None = None
    stop_price: float | None = None
    zone_id: str | None = None
    ladder_id: str | None = None
    expires_at: str | None = None
    created_at: str = field(default_factory=utc_iso)
    id: str = field(default_factory=lambda: new_id("tp"))

    @property
    def notional_usd(self) -> float:
        return self.price * self.quantity


@dataclass(slots=True)
class TradeIntent:
    """LLM-facing proposal type.

    Agents may emit this structure, but only deterministic risk and execution
    services can turn it into exchange orders.
    """

    symbol: str
    side: Side
    limit_price: float
    quantity: float
    confidence: float
    expected_edge_bps: float
    stop_loss_pct: float
    take_profit_pct: float
    rationale: str
    evidence_ids: list[str]
    risk_bps: float | None = None
    source_agent: str = "strategy"
    # Demand-zone laddered entry context; see TradeProposal for field meanings.
    target_price: float | None = None
    stop_price: float | None = None
    zone_id: str | None = None
    ladder_id: str | None = None
    expires_at: str | None = None
    created_at: str = field(default_factory=utc_iso)
    id: str = field(default_factory=lambda: new_id("ti"))

    @property
    def notional_usd(self) -> float:
        return self.limit_price * self.quantity

    def to_proposal(self) -> TradeProposal:
        proposal_id = self.id.replace("ti_", "tp_", 1)
        return TradeProposal(
            id=proposal_id,
            symbol=self.symbol,
            side=self.side,
            price=self.limit_price,
            quantity=self.quantity,
            confidence=self.confidence,
            expected_edge_bps=self.expected_edge_bps,
            risk_bps=self.risk_bps if self.risk_bps is not None else self.stop_loss_pct * 10_000,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            rationale=self.rationale,
            evidence_ids=self.evidence_ids,
            target_price=self.target_price,
            stop_price=self.stop_price,
            zone_id=self.zone_id,
            ladder_id=self.ladder_id,
            expires_at=self.expires_at,
            created_at=self.created_at,
        )

    @classmethod
    def from_proposal(cls, proposal: TradeProposal, source_agent: str = "strategy") -> TradeIntent:
        return cls(
            symbol=proposal.symbol,
            side=proposal.side,
            limit_price=proposal.price,
            quantity=proposal.quantity,
            confidence=proposal.confidence,
            expected_edge_bps=proposal.expected_edge_bps,
            risk_bps=proposal.risk_bps,
            stop_loss_pct=proposal.stop_loss_pct,
            take_profit_pct=proposal.take_profit_pct,
            rationale=proposal.rationale,
            evidence_ids=proposal.evidence_ids,
            target_price=proposal.target_price,
            stop_price=proposal.stop_price,
            zone_id=proposal.zone_id,
            ladder_id=proposal.ladder_id,
            expires_at=proposal.expires_at,
            source_agent=source_agent,
            created_at=proposal.created_at,
        )


@dataclass(slots=True)
class RiskDecision:
    proposal_id: str
    approved: bool
    reasons: list[str]
    created_at: str = field(default_factory=utc_iso)
    id: str = field(default_factory=lambda: new_id("rd"))


@dataclass(slots=True)
class PriceLevel:
    """A single candidate price level from one detection method on one timeframe.

    ``weight`` is the method-x-timeframe contribution that gets summed when nearby
    levels are clustered into a SupportZone (higher-timeframe, higher-conviction
    methods carry more). ``kind`` is the detector: swing_low/swing_high, hvn
    (volume node), prior_high/prior_low/prior_close, fib, round, ema.
    """

    price: float
    kind: str
    timeframe: str
    weight: float
    label: str = ""


@dataclass(slots=True)
class SupportZone:
    """A clustered price band (not a single line) where multiple methods agree.

    ``side`` is relative to the price the map was built at: a zone whose mid sits
    below price is a "support" (demand) zone to bid; above is "resistance" (a
    target). ``strength`` is the summed confluence weight; ``methods``/``timeframes``
    record what produced it for auditability. ``distance_pct`` is signed: negative
    means the zone is below the current price.
    """

    low: float
    high: float
    mid: float
    strength: float
    side: str
    methods: list[str]
    timeframes: list[str]
    distance_pct: float
    touches: int = 0
    id: str = field(default_factory=lambda: new_id("zone"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LevelMap:
    """Per-symbol support/resistance picture for one cycle.

    Support zones are sorted nearest-below-first, resistance nearest-above-first,
    so ``nearest_support()`` is the first zone a falling price would reach. Built
    deterministically by core.levels from monthly..1h candles; a technical_analyst
    agent confirms which zone to actually bid.
    """

    symbol: str
    current_price: float
    regime: str
    support_zones: list[SupportZone]
    resistance_zones: list[SupportZone]
    created_at: str = field(default_factory=utc_iso)

    def nearest_support(self) -> SupportZone | None:
        return self.support_zones[0] if self.support_zones else None

    def nearest_resistance(self) -> SupportZone | None:
        return self.resistance_zones[0] if self.resistance_zones else None

    def zone_by_id(self, zone_id: str) -> SupportZone | None:
        for zone in (*self.support_zones, *self.resistance_zones):
            if zone.id == zone_id:
                return zone
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "regime": self.regime,
            "support_zones": [zone.to_dict() for zone in self.support_zones],
            "resistance_zones": [zone.to_dict() for zone in self.resistance_zones],
            "created_at": self.created_at,
        }


# Deterministic per-order recommendations the team reacts to (see PositionReview).
# HOLD/ADJUST/CLOSE apply to a filled position; KEEP/CANCEL to a resting bid.
REVIEW_HOLD = "HOLD"
REVIEW_ADJUST = "ADJUST"
REVIEW_CLOSE_CANDIDATE = "CLOSE_CANDIDATE"
REVIEW_KEEP = "KEEP"
REVIEW_CANCEL_CANDIDATE = "CANCEL_CANDIDATE"


@dataclass(slots=True)
class PositionReview:
    """Deterministic review of one OPEN order (filled position or resting bid).

    The position-review counterpart to a TradeProposal: prepared each cycle so the
    agent team reviews open risk FIRST, starting from a concrete recommended action
    rather than free-form. ``recommended_action`` is conservative (defaults to
    HOLD/KEEP and never loosens a stop); the team confirms or overrides it.
    ``min_hold_satisfied`` reflects the anti-churn cooldown — a discretionary CLOSE
    is blocked while False (the deterministic stop still protects regardless).
    """

    order_id: str
    symbol: str
    status: str  # OrderStatus value: ENTRY_OPEN (resting bid) | POSITION_OPEN
    recommended_action: str
    reason: str
    age_minutes: float | None
    min_hold_satisfied: bool
    regime: str | None = None
    current_price: float | None = None
    unrealized_pnl_pct: float | None = None
    to_take_profit_pct: float | None = None
    to_stop_loss_pct: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    @property
    def is_resting_bid(self) -> bool:
        return self.status == OrderStatus.ENTRY_OPEN.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExitLeg:
    """One take-profit tier in a scale-out ladder.

    ``size_pct`` is a fraction of the order's ORIGINAL filled quantity. The leg
    sells that fraction once price reaches ``target_price``; ``filled`` flips
    once its partial sell is confirmed.
    """

    tier: int
    target_price: float
    size_pct: float
    filled: bool = False
    filled_qty: float = 0.0
    filled_at: str | None = None
    exit_order_exchange_id: str | None = None
    exit_client_order_id: str | None = None


@dataclass(slots=True)
class ExitPlan:
    """Deterministic tiered exit attached to an open position.

    The TP legs scale out partial size on the way up; the runner (the unsold
    remainder) has no fixed target and rides ``current_stop_price``, which only
    ever moves up (breakeven after an early tier, then locked to a prior tier,
    then trailing the high-water mark). ``tiered=False`` is the legacy
    single-leg bracket (one TP leg at 100%, fixed stop) for backward compat.
    """

    legs: list[ExitLeg]
    initial_stop_price: float
    current_stop_price: float
    high_water_price: float
    runner_size_pct: float = 0.0
    tiered: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExitPlan:
        legs = [ExitLeg(**leg) for leg in data.get("legs", [])]
        return cls(
            legs=legs,
            initial_stop_price=float(data["initial_stop_price"]),
            current_stop_price=float(data["current_stop_price"]),
            high_water_price=float(data["high_water_price"]),
            runner_size_pct=float(data.get("runner_size_pct", 0.0)),
            tiered=bool(data.get("tiered", True)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> ExitPlan | None:
        if not raw:
            return None
        return cls.from_dict(json.loads(raw))

    @property
    def runner_active(self) -> bool:
        """True once every TP leg has filled and only the runner remains."""
        return bool(self.runner_size_pct > 0) and all(leg.filled for leg in self.legs)

    @property
    def tiers_filled(self) -> int:
        return sum(1 for leg in self.legs if leg.filled)


@dataclass(slots=True)
class OrderRecord:
    proposal_id: str
    mode: str
    symbol: str
    side: Side
    order_type: str
    price: float
    quantity: float
    take_profit_price: float
    stop_loss_price: float
    status: OrderStatus = OrderStatus.ENTRY_OPEN
    opened_at: str = field(default_factory=utc_iso)
    closed_at: str | None = None
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    realized_pnl: float = 0.0
    exit_price: float | None = None
    exit_reason: str | None = None
    # Exchange truth (testnet/live): the local row is a cache of the venue's
    # view, refreshed by ExchangeReconciler; these stay None/0 until the exchange reports fills.
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    exchange_status: str | None = None
    executed_qty: float = 0.0
    cumulative_quote_qty: float = 0.0
    avg_fill_price: float | None = None
    commission_total: float = 0.0
    commission_asset: str | None = None
    pnl_estimated: bool = False
    last_synced_at: str | None = None
    decision_id: str | None = None
    closed_by: str | None = None
    exit_order_exchange_id: str | None = None
    exit_client_order_id: str | None = None
    # Tiered scale-out ladder (TP1/TP2/runner + ratcheting stop). None for
    # legacy rows; synthesized as a single-leg plan on load when absent.
    exit_plan: ExitPlan | None = None
    id: str = field(default_factory=lambda: new_id("ord"))


@dataclass(slots=True)
class FillRecord:
    """One executed trade (fill) reported by the exchange for an order."""

    order_id: str
    symbol: str
    side: Side
    price: float
    qty: float
    quote_qty: float
    commission: float
    commission_asset: str | None
    exchange_order_id: str | None = None
    exchange_trade_id: int | None = None
    is_exit: bool = False
    trade_time: str | None = None
    raw_json: str | None = None
    id: str = field(default_factory=lambda: new_id("fill"))


def to_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def dataclass_json(data: Any) -> str:
    return to_json(asdict(data))
