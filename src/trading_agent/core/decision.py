"""Supervisor trade decisions: schema, parsing, and deterministic fallback.

The supervisor deep agent is the trader: after consulting all specialist
subagents it must emit one final fenced ```json block containing its
decision(s). That block is parsed and validated here. Anything malformed,
incomplete, or missing degrades deterministically to WAIT — the LLM can
never bypass the risk gate by producing creative output. Approved BUY
decisions map onto TradeIntent and flow through the unchanged deterministic
pre-trade checks and RiskGovernor.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from trading_agent.core.config import AppConfig
from trading_agent.core.models import Side, TradeIntent, new_id, utc_iso

REQUIRED_AGENTS = frozenset(
    {
        "market_research",
        "technical_analyst",
        "news_research",
        "onchain_research",
        "strategy",
        "risk_review",
        "reporting",
    }
)

# Capture the whole fenced block body; JSON nesting makes brace-matching
# regexes unreliable, so json.loads validates the content afterwards.
_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

_ACTION_ALIASES = {
    "buy": "BUY",
    "long": "BUY",
    "open": "BUY",
    "sell": "SELL",
    "short": "SELL",
    "wait": "WAIT",
    "hold": "WAIT",
    "no_trade": "WAIT",
    "no trade": "WAIT",
    "stand_aside": "WAIT",
    "stand aside": "WAIT",
    "close": "CLOSE",
    "exit": "CLOSE",
    "adjust": "ADJUST",
    "update": "ADJUST",
}

_STANCE_ALIASES = {
    "bullish": "bullish",
    "buy": "bullish",
    "long": "bullish",
    "support_buy": "bullish",
    "approve_buy": "bullish",
    "approve_long": "bullish",
    "support_long": "bullish",
    "favor_long": "bullish",
    "slightly_bullish": "bullish",
    "mildly_bullish": "bullish",
    "positive": "bullish",
    "bearish": "bearish",
    "sell": "bearish",
    "short": "bearish",
    "support_sell": "bearish",
    "support_close": "bearish",
    "approve_short": "bearish",
    "support_short": "bearish",
    "reject_long": "bearish",
    "reject_buy": "bearish",
    "oppose_long": "bearish",
    "reject_short": "neutral",
    "reject_sell": "neutral",
    "close": "bearish",
    "exit": "bearish",
    "slightly_bearish": "bearish",
    "mildly_bearish": "bearish",
    "negative": "bearish",
    "neutral": "neutral",
    "hold": "neutral",
    "wait": "neutral",
    "prefer_wait": "neutral",
    "no_trade": "neutral",
    "no trade": "neutral",
    "flat": "neutral",
    "mixed": "neutral",
    "balanced": "neutral",
    "sideways": "neutral",
    "rangebound": "neutral",
    "cautious": "neutral",
    "cautious_approve": "neutral",
    "low_conviction": "neutral",
    "low-conviction": "neutral",
    "abstain": "abstain",
    "unknown": "abstain",
    "n/a": "abstain",
    "na": "abstain",
    "none": "abstain",
    "no_view": "abstain",
    "no view": "abstain",
}


class DecisionAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    CLOSE = "CLOSE"
    ADJUST = "ADJUST"


class AgentConsultation(BaseModel):
    agent: str
    stance: Literal["bullish", "bearish", "neutral", "abstain"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    # Binding sizing lever for risk_review: a 0..1 multiplier the deterministic
    # sizer applies to a BUY's notional (1.0 = no change, 0.0 = veto). Ignored for
    # other agents. Lets the pre-mortem actually shrink/veto a trade instead of
    # only annotating it. Defaults to 1.0 so existing output is unaffected.
    size_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)


class SupervisorDecision(BaseModel):
    action: DecisionAction
    symbol: str
    limit_price: float | None = None
    quantity: float | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    # Absolute demand-zone levels. When the desk reasons in prices ("invalidation
    # 72.08, target 73.84") it can give them here; they take precedence over the
    # percentages so the BOOKED stop is the zone invalidation, not a fixed % off
    # entry. Either form works; the gate reconciles them.
    stop_price: float | None = None
    target_price: float | None = None
    target_order_id: str | None = None
    new_take_profit_price: float | None = None
    new_stop_loss_price: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_edge_bps: float = 0.0
    rationale: str = ""
    consultations: list[AgentConsultation] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source: str = "supervisor"
    created_at: str = Field(default_factory=utc_iso)
    id: str = Field(default_factory=lambda: new_id("dec"))

    @model_validator(mode="after")
    def _validate_by_action(self) -> SupervisorDecision:
        if self.action == DecisionAction.BUY:
            if not self.limit_price or self.limit_price <= 0:
                raise ValueError("BUY requires a positive limit_price")
            if not self.quantity or self.quantity <= 0:
                raise ValueError("BUY requires a positive quantity")
            consulted = {c.agent for c in self.consultations}
            missing = sorted(REQUIRED_AGENTS - consulted)
            if missing:
                raise ValueError(
                    "BUY requires consultations from all agents; missing: " + ", ".join(missing)
                )
        if self.action in {DecisionAction.SELL, DecisionAction.CLOSE} and not self.target_order_id:
            raise ValueError(f"{self.action.value} requires target_order_id of the open order")
        if self.action == DecisionAction.ADJUST:
            if not self.target_order_id:
                raise ValueError("ADJUST requires target_order_id of the open order")
            if self.new_take_profit_price is None and self.new_stop_loss_price is None:
                raise ValueError("ADJUST requires new_take_profit_price and/or new_stop_loss_price")
        return self

    def to_intent(self, config: AppConfig) -> TradeIntent:
        if self.action != DecisionAction.BUY:
            raise ValueError("only BUY decisions map to TradeIntent")
        limit = float(self.limit_price or 0.0)
        stop_pct = self.stop_loss_pct if self.stop_loss_pct is not None else config.risk.stop_loss_pct
        tp_pct = (
            self.take_profit_pct if self.take_profit_pct is not None else config.risk.take_profit_pct
        )
        # Resolve ABSOLUTE stop/target so the booked exit bracket equals the zone
        # invalidation the desk reasoned. Prefer an explicit price; otherwise derive
        # it from the percentage. Keep the relative pct consistent with the absolute
        # stop so the RiskGovernor's per-trade risk check matches the booked stop.
        stop_price = self.stop_price
        if stop_price is not None and stop_price > 0 and limit > 0:
            stop_pct = max(0.0, (limit - stop_price) / limit)
        elif limit > 0:
            stop_price = round(limit * (1 - stop_pct), 8)
        target_price = self.target_price
        if target_price is not None and target_price > 0 and limit > 0:
            tp_pct = max(0.0, (target_price / limit) - 1)
        elif limit > 0:
            target_price = round(limit * (1 + tp_pct), 8)
        return TradeIntent(
            symbol=self.symbol,
            side=Side.BUY,
            limit_price=limit,
            quantity=float(self.quantity or 0.0),
            confidence=self.confidence,
            expected_edge_bps=self.expected_edge_bps,
            stop_loss_pct=stop_pct,
            take_profit_pct=tp_pct,
            rationale=self.rationale,
            evidence_ids=list(self.evidence_refs),
            target_price=target_price,
            stop_price=stop_price,
            source_agent=self.source,
        )

    def summary_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def wait_decision(symbol: str, reason: str, *, source: str = "supervisor") -> SupervisorDecision:
    return SupervisorDecision(
        action=DecisionAction.WAIT,
        symbol=symbol,
        rationale=reason,
        source=source,
    )


def parse_supervisor_decisions(
    text: str, *, source: str = "supervisor"
) -> tuple[list[SupervisorDecision], list[str]]:
    """Extract and validate decision JSON from supervisor output.

    Accepts a single object, a list of objects, or {"decisions": [...]} inside
    the LAST fenced code block that mentions an "action". Returns
    (valid decisions, human-readable errors for everything rejected).
    """
    if not text:
        return [], ["supervisor produced no output"]
    candidates = [match.group(1).strip() for match in _JSON_BLOCK.finditer(text)]
    decision_blocks = [c for c in candidates if '"action"' in c and c.startswith(("{", "["))]
    if not decision_blocks:
        return [], ["no decision JSON block found in supervisor output"]
    raw_block = decision_blocks[-1]
    try:
        parsed = json.loads(raw_block)
    except json.JSONDecodeError as exc:
        return [], [f"decision block is not valid JSON: {exc}"]
    if isinstance(parsed, dict) and isinstance(parsed.get("decisions"), list):
        items: list[Any] = parsed["decisions"]
    elif isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return [], ["decision block must be a JSON object or array"]

    decisions: list[SupervisorDecision] = []
    errors: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"decision[{index}] is not a JSON object")
            continue
        item = dict(item)
        _normalize_decision_item(item)
        item["source"] = source
        item.pop("id", None)
        item.pop("created_at", None)
        try:
            decisions.append(SupervisorDecision.model_validate(item))
        except ValidationError as exc:
            symbol = str(item.get("symbol", "?"))
            first = exc.errors()[0]
            errors.append(f"decision[{index}] ({symbol}): {first.get('msg', 'invalid')}")
    return decisions, errors


def _normalize_decision_item(item: dict[str, Any]) -> None:
    action = _normalize_alias(item.get("action"), _ACTION_ALIASES)
    if action is not None:
        item["action"] = action

    consultations = item.get("consultations")
    if not isinstance(consultations, list):
        return
    for consultation in consultations:
        if not isinstance(consultation, dict):
            continue
        raw_stance = consultation.get("stance")
        stance = _normalize_alias(raw_stance, _STANCE_ALIASES)
        if stance is None:
            stance = _infer_stance(raw_stance)
        consultation["stance"] = stance


_NEGATION_TOKENS = ("reject", "oppose", "against", "avoid", "veto", "block", "deny", "decline")
_BULLISH_TOKENS = ("bull", "long", "buy", "accumulate", "overweight")
_BEARISH_TOKENS = ("bear", "short", "sell", "underweight")
_NEUTRAL_TOKENS = ("neutral", "hold", "wait", "flat", "mixed")


def _infer_stance(value: Any) -> str:
    """Best-effort stance for vocabulary outside _STANCE_ALIASES.

    A consultation stance the schema does not recognize must never reject the
    whole decision (observed live: risk_review said "approve_long" and the
    supervisor's BUY was dropped). Unrecognizable wording degrades to abstain.
    """
    if not isinstance(value, str):
        return "abstain"
    normalized = re.sub(r"[^a-z]+", "_", value.strip().lower()).strip("_")
    negated = any(token in normalized for token in _NEGATION_TOKENS)
    bullish = any(token in normalized for token in _BULLISH_TOKENS)
    bearish = any(token in normalized for token in _BEARISH_TOKENS)
    if bullish and not bearish:
        return "bearish" if negated else "bullish"
    if bearish and not bullish:
        return "neutral" if negated else "bearish"
    if any(token in normalized for token in _NEUTRAL_TOKENS):
        return "neutral"
    return "abstain"


def _normalize_alias(value: Any, aliases: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized in aliases:
        return aliases[normalized]
    compact = re.sub(r"[^a-z_ ]+", "", normalized)
    return aliases.get(compact)


def decisions_from_intents(
    intents: list[TradeIntent], *, source: str = "deterministic"
) -> list[SupervisorDecision]:
    """Wrap deterministic StrategyAgent output so the graph shape stays uniform
    when the LLM supervisor is disabled."""
    decisions: list[SupervisorDecision] = []
    for intent in intents:
        consultations = [
            AgentConsultation(
                agent=agent,
                stance="neutral",
                confidence=intent.confidence,
                summary="deterministic pipeline (LLM supervisor disabled)",
            )
            for agent in sorted(REQUIRED_AGENTS)
        ]
        decisions.append(
            SupervisorDecision(
                action=DecisionAction.BUY,
                symbol=intent.symbol,
                limit_price=intent.limit_price,
                quantity=intent.quantity,
                stop_loss_pct=intent.stop_loss_pct,
                take_profit_pct=intent.take_profit_pct,
                stop_price=intent.stop_price,
                target_price=intent.target_price,
                confidence=intent.confidence,
                expected_edge_bps=intent.expected_edge_bps,
                rationale=intent.rationale,
                consultations=consultations,
                evidence_refs=list(intent.evidence_ids),
                source=source,
            )
        )
    return decisions


DECISION_FORMAT_REFERENCE = """
FINAL DECISION CONTRACT (mandatory)
After consulting ALL seven subagents (market_research, technical_analyst,
news_research, onchain_research, strategy, risk_review, reporting), end your reply
with ONE fenced json block containing your decision(s) — an object, or an array
with one object per symbol:

```json
{
  "action": "BUY | SELL | WAIT | CLOSE | ADJUST",
  "symbol": "BTCUSDT",
  "limit_price": 62500.0,           // BUY: required; rest in/at the demand zone BELOW the bid, never chase above it
  "quantity": 0.0016,               // BUY: required; ~100 USD notional
  "stop_loss_pct": 0.025,           // BUY: optional; place the stop just BELOW the demand zone
  "take_profit_pct": 0.06,          // BUY: optional; target the next resistance for >= 1.5R
  "stop_price": 62300.0,            // BUY: optional; ABSOLUTE invalidation just below the zone (preferred over stop_loss_pct; this is the booked stop)
  "target_price": 67310.0,         // BUY: optional; ABSOLUTE target at the next resistance (preferred over take_profit_pct)
  "target_order_id": "ord_abc123",  // CLOSE/SELL/ADJUST: required (the open order's id)
  "new_take_profit_price": null,    // ADJUST only
  "new_stop_loss_price": null,      // ADJUST only
  "confidence": 0.74,
  "expected_edge_bps": 600.0,       // geometric: distance from this entry to the target, in bps
  "rationale": "one or two sentences citing the decisive evidence + the demand zone",
  "consultations": [                // BUY: must include ALL seven agents
    {"agent": "market_research", "stance": "bullish", "confidence": 0.8, "summary": "..."},
    {"agent": "technical_analyst", "stance": "bullish", "confidence": 0.75, "summary": "demand zone X-Y, invalidation below Z"},
    {"agent": "news_research", "stance": "neutral", "confidence": 0.6, "summary": "..."},
    {"agent": "onchain_research", "stance": "bullish", "confidence": 0.7, "summary": "..."},
    {"agent": "strategy", "stance": "bullish", "confidence": 0.75, "summary": "..."},
    {"agent": "risk_review", "stance": "neutral", "confidence": 0.7, "summary": "..."},
    {"agent": "reporting", "stance": "abstain", "confidence": 0.5, "summary": "..."}
  ],
  "evidence_refs": ["ev_..."]
}
```

Rules:
- WAIT is acceptable, but PREFER a resting demand-zone BUY when the read is not
  bearish: instead of WAITing because price is poorly located, bid the next
  technical_analyst-confirmed support zone (well below price) with the stop just
  below that zone. A bid that does not fill for several cycles is fine and expected.
- A BUY without all seven consultations, or with a price you did not verify via
  get_price/get_orderbook_ticker, will be rejected and treated as WAIT.
- evidence_refs MUST be ids copied verbatim from the CYCLE CONTEXT
  "evidence_available" list (e.g. "ev_..."). A BUY whose evidence_refs cite no
  real id is rejected — do not invent ids or describe evidence in prose there.
- Malformed JSON is treated as WAIT for every symbol. Your decision is then
  checked by the deterministic RiskGovernor, which can veto it.
""".strip()
