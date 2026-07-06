from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from trading_agent.core.config import AppConfig, StrategyConfig
from trading_agent.core.models import (
    EvidenceRecord,
    LevelMap,
    MarketSnapshot,
    Side,
    TradeProposal,
    new_id,
    utc_now,
)
from trading_agent.core.risk import conviction_size, maker_pullback_price


# Default weights, kept for callers that score evidence outside a full AppConfig.
# The live values come from config.strategy (see StrategyConfig).
AGENT_WEIGHTS = StrategyConfig().agent_weights

# Evidence this bearish (combined, weighted) means the desk does not accumulate the
# name at all — a demand zone is only worth bidding when the read is not negative.
_BEARISH_SKIP_SCORE = -0.10


class StrategyAgent:
    name = "strategy_decision_agent"

    def propose(
        self,
        snapshots: dict[str, MarketSnapshot],
        evidence: list[EvidenceRecord],
        config: AppConfig,
        level_maps: dict[str, LevelMap] | None = None,
    ) -> list[TradeProposal]:
        grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in evidence:
            grouped[record.symbol].append(record)

        tuning = config.strategy
        level_maps = level_maps or {}
        proposals: list[TradeProposal] = []
        for symbol, records in grouped.items():
            snapshot = snapshots[symbol]
            # Placeholder evidence is synthesized noise emitted when live
            # sources are down; it must never move a trade decision.
            live_records = [record for record in records if not record.is_placeholder]
            if not live_records:
                continue
            weighted_score = 0.0
            weight_total = 0.0
            confidence_total = 0.0
            for record in live_records:
                # Honest down-weighting: a degraded fallback source (web news,
                # coarse TVL proxy) counts for less than a primary source. quality
                # is 1.0 for primary, lower for fallbacks (see EvidenceRecord).
                weight = tuning.agent_weights.get(record.agent, tuning.default_agent_weight) * record.quality
                weighted_score += record.score * weight
                weight_total += weight
                confidence_total += record.confidence * record.quality
            if weight_total == 0:
                continue
            combined_score = weighted_score / weight_total
            avg_agent_confidence = confidence_total / len(live_records)
            expected_edge_bps = combined_score * tuning.edge_scale_bps
            proposal_confidence = min(
                tuning.confidence_cap,
                tuning.confidence_base
                + abs(combined_score) * tuning.confidence_score_coef
                + avg_agent_confidence * tuning.confidence_agreement_coef,
            )
            evidence_ids = [record.id for record in live_records]

            # Preferred path: rest scaled limit bids INTO the next computed demand
            # zone(s) instead of WAITing because price is poorly located. Requires a
            # level map and a read that is not bearish.
            level_map = level_maps.get(symbol)
            if (
                config.risk.require_zone_anchored_bids
                and level_map is not None
                and level_map.support_zones
                and combined_score >= _BEARISH_SKIP_SCORE
            ):
                ladder = self._zone_ladder(
                    symbol=symbol,
                    snapshot=snapshot,
                    level_map=level_map,
                    confidence=proposal_confidence,
                    evidence_ids=evidence_ids,
                    config=config,
                )
                if ladder:
                    proposals.extend(ladder)
                    continue
                # No zone cleared the reward:risk bar -> fall through to the legacy
                # shallow maker entry (which may itself skip on edge).

            # Legacy shallow maker-pullback entry (used when no level map is
            # available, e.g. offline backtests, or no zone qualified).
            if expected_edge_bps < config.risk.min_expected_edge_bps:
                continue
            limit_price = maker_pullback_price(snapshot, config.risk)
            atr_pct = (snapshot.atr / snapshot.last_price) if (snapshot.atr and snapshot.last_price) else None
            target_notional = conviction_size(
                confidence=proposal_confidence,
                expected_edge_bps=expected_edge_bps,
                atr_pct=atr_pct,
                budget_usd=config.live.capital_budget_usd,
                sizing=config.sizing,
            )
            if target_notional <= 0 or limit_price <= 0:
                continue
            quantity = target_notional / limit_price
            rationale = (
                f"Combined score {combined_score:.3f} from {len(live_records)} live "
                f"evidence record(s); expected edge {expected_edge_bps:.1f} bps."
            )
            proposals.append(
                TradeProposal(
                    symbol=symbol,
                    side=Side.BUY,
                    price=round(limit_price, 8),
                    quantity=round(quantity, 8),
                    confidence=round(proposal_confidence, 6),
                    expected_edge_bps=round(expected_edge_bps, 4),
                    risk_bps=round(config.risk.stop_loss_pct * 10_000, 4),
                    stop_loss_pct=config.risk.stop_loss_pct,
                    take_profit_pct=config.risk.take_profit_pct,
                    rationale=rationale,
                    evidence_ids=evidence_ids,
                )
            )
        proposals.sort(key=lambda item: (item.confidence, item.expected_edge_bps), reverse=True)
        return proposals

    def _zone_ladder(
        self,
        *,
        symbol: str,
        snapshot: MarketSnapshot,
        level_map: LevelMap,
        confidence: float,
        evidence_ids: list[str],
        config: AppConfig,
    ) -> list[TradeProposal]:
        """Build scaled resting bids into demand zones with stops below each zone.

        Returns one TradeProposal per ladder leg (front-loaded size), each carrying
        absolute target/stop, the zone id, a shared ladder id, and an expiry. Returns
        [] if no zone clears the reward:risk bar or the read/regime forbids bidding.
        """
        risk = config.risk
        price = snapshot.last_price or level_map.current_price
        bid = snapshot.bid_price or price
        if price <= 0:
            return []

        # Target for the geometric edge: the nearest resistance zone above price,
        # else a default swing target off the configured take_profit width.
        nearest_res = level_map.nearest_resistance()
        default_target = price * (1 + risk.take_profit_pct)
        target_price = nearest_res.mid if (nearest_res and nearest_res.mid > price) else default_target

        downtrend = level_map.regime == "downtrend"
        candidate_zones = list(level_map.support_zones)
        if downtrend:
            # Knife-catching guard: in a confirmed downtrend work only the single
            # deepest, highest-confluence zone, at reduced size.
            candidate_zones.sort(key=lambda z: (z.strength, price - z.mid), reverse=True)
            candidate_zones = candidate_zones[:1]

        legs: list[dict] = []
        for zone in candidate_zones:
            entry = round(min(zone.high, bid), 8)
            if entry <= 0:
                continue
            depth = (price - entry) / price
            if depth > risk.max_bid_depth_pct or depth < 0:
                continue
            stop = zone.low * (1 - risk.zone_stop_buffer_pct)
            if stop <= 0 or stop >= entry:
                continue
            leg_target = max(target_price, entry * (1 + risk.take_profit_pct))
            stop_loss_pct = (entry - stop) / entry
            take_profit_pct = (leg_target - entry) / entry
            if stop_loss_pct <= 0 or take_profit_pct <= 0:
                continue
            reward_risk = take_profit_pct / stop_loss_pct
            if reward_risk < risk.min_reward_risk:
                continue
            legs.append(
                {
                    "zone": zone,
                    "entry": entry,
                    "stop": round(stop, 8),
                    "target": round(leg_target, 8),
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "edge_bps": take_profit_pct * 10_000,
                    "reward_risk": reward_risk,
                }
            )
            if len(legs) >= max(1, risk.ladder_legs):
                break

        if not legs:
            return []

        # Size the whole ladder once (conviction x best geometric edge), de-risked in
        # a downtrend, then split front-loaded across the legs.
        best_edge = max(leg["edge_bps"] for leg in legs)
        atr_pct = (snapshot.atr / price) if (snapshot.atr and price) else None
        regime_mult = risk.downtrend_size_mult if downtrend else 1.0
        total_notional = conviction_size(
            confidence=confidence,
            expected_edge_bps=best_edge,
            atr_pct=atr_pct,
            budget_usd=config.live.capital_budget_usd,
            sizing=config.sizing,
            regime_mult=regime_mult,
        )
        if total_notional <= 0:
            return []

        # Cap legs so each carries at least the exchange minimum notional (no dust).
        min_notional = config.sizing.min_notional_usd
        max_legs = max(1, int(total_notional // min_notional))
        legs = legs[: min(len(legs), max_legs)]

        weights = list(risk.ladder_size_split) or [1.0]
        leg_weights = [weights[min(i, len(weights) - 1)] for i in range(len(legs))]
        weight_sum = sum(leg_weights) or 1.0

        ladder_id = new_id("ladder")
        expires_at = None
        if risk.bid_ttl_minutes > 0:
            expires_at = (utc_now() + timedelta(minutes=risk.bid_ttl_minutes)).isoformat()

        out: list[TradeProposal] = []
        for leg, weight in zip(legs, leg_weights):
            leg_notional = total_notional * (weight / weight_sum)
            if leg_notional < min_notional:
                continue
            entry = leg["entry"]
            quantity = leg_notional / entry
            zone = leg["zone"]
            rationale = (
                f"Demand-zone bid ({level_map.regime}): rest {entry} into support "
                f"{zone.low}-{zone.high} [{','.join(zone.methods)}], stop {leg['stop']} "
                f"below zone, target {leg['target']} = {leg['reward_risk']:.1f}R."
            )
            out.append(
                TradeProposal(
                    symbol=symbol,
                    side=Side.BUY,
                    price=round(entry, 8),
                    quantity=round(quantity, 8),
                    confidence=round(confidence, 6),
                    expected_edge_bps=round(leg["edge_bps"], 4),
                    risk_bps=round(leg["stop_loss_pct"] * 10_000, 4),
                    stop_loss_pct=round(leg["stop_loss_pct"], 8),
                    take_profit_pct=round(leg["take_profit_pct"], 8),
                    rationale=rationale,
                    evidence_ids=evidence_ids,
                    target_price=leg["target"],
                    stop_price=leg["stop"],
                    zone_id=zone.id,
                    ladder_id=ladder_id,
                    expires_at=expires_at,
                )
            )
        return out
