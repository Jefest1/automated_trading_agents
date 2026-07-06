from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol

from trading_agent.core.logging import get_logger
from trading_agent.core.models import EvidenceRecord, MarketSnapshot
from trading_agent.utils.binance_skills import MAJOR_TOKEN_CONTRACTS, BinanceSkillRegistry
from trading_agent.utils.free_feeds import (
    binance_derivatives,
    crypto_news_rss,
    defillama_tvl_change,
    gdelt_headlines,
)
from trading_agent.utils.web_search import run_web_news_search

LOGGER = get_logger("signals")

# Leaderboard-style skill commands (social-hype, smart-money-inflow) return
# one chain-wide list, so a single call serves every symbol in a cycle.
_SKILL_CACHE_TTL_SECONDS = 300.0
_skill_cache: dict[str, tuple[float, Any]] = {}

_POSITIVE_WORDS = (
    "surge", "rally", "bullish", "gain", "soar", "record", "approval", "adoption",
    "inflow", "upgrade", "partnership", "breakout", "all-time high", "buy",
)
_NEGATIVE_WORDS = (
    "crash", "plunge", "bearish", "drop", "hack", "exploit", "lawsuit", "ban",
    "outflow", "downgrade", "sell-off", "liquidation", "fraud", "fear",
)
_TOKEN_NAMES = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "bnb binance coin",
}

# Derivatives-positioning thresholds (Binance funding is per 8h). Tunable.
_FUNDING_SATURATION = 0.0005  # ~0.05%/8h -> strong net positioning (score ~1)
_FUNDING_EXTREME = 0.001  # >0.1%/8h = overcrowded -> fade toward contrarian
_OI_CHANGE_SATURATION = 0.05  # ~5% recent OI change -> strong conviction shift


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class SignalAgent(Protocol):
    name: str

    def analyze(self, symbol: str, snapshot: MarketSnapshot, cycle: int) -> EvidenceRecord:
        ...


class MarketFeed(Protocol):
    def snapshot(self, symbol: str, cycle: int) -> MarketSnapshot:
        ...


class MarketDataAgent:
    name = "market_data_agent"
    source = "binance-compatible-feed"

    def __init__(self, feed: MarketFeed) -> None:
        self.feed = feed
        self._previous: dict[str, MarketSnapshot] = {}

    def analyze(self, symbol: str, snapshot: MarketSnapshot, cycle: int) -> EvidenceRecord:
        previous = self._previous.get(symbol) or self.feed.snapshot(symbol, max(0, cycle - 1))
        self._previous[symbol] = snapshot
        momentum_bps = ((snapshot.last_price / previous.last_price) - 1) * 10_000
        spread_bps = ((snapshot.ask_price - snapshot.bid_price) / snapshot.last_price) * 10_000
        score = clamp((momentum_bps - spread_bps * 0.2) / 80.0)
        confidence = clamp(0.62 + abs(score) * 0.28, 0.0, 0.95)
        return EvidenceRecord(
            agent=self.name,
            source=self.source,
            symbol=symbol,
            kind="price_order_book",
            observed_at=snapshot.observed_at,
            score=round(score, 6),
            confidence=round(confidence, 6),
            payload={
                "last_price": snapshot.last_price,
                "bid_price": snapshot.bid_price,
                "ask_price": snapshot.ask_price,
                "volume_24h": snapshot.volume_24h,
                "momentum_bps": round(momentum_bps, 4),
                "spread_bps": round(spread_bps, 4),
            },
        )


class NewsSentimentAgent:
    """News sentiment, free-first.

    Source order: Binance Skills Hub social-hype -> live web news headlines ->
    GDELT last-24h articles -> deterministic placeholder. The source actually
    used is recorded in the evidence payload so every score is auditable.
    """

    name = "news_sentiment_agent"

    def __init__(
        self,
        registry: BinanceSkillRegistry | None = None,
        *,
        enable_web_news: bool = False,
    ) -> None:
        self.registry = registry
        self.enable_web_news = enable_web_news

    def analyze(self, symbol: str, snapshot: MarketSnapshot, cycle: int) -> EvidenceRecord:
        token = symbol.replace("USDT", "")
        # Crypto-specific RSS headlines first (keyless, reliable), then Binance
        # social-hype, then generic web/GDELT, then a deterministic placeholder.
        scored = self._score_from_rss(token) if self.enable_web_news else None
        if scored is None:
            scored = self._score_from_skills(token)
        if scored is None and self.enable_web_news:
            scored = self._score_from_web_news(token)
        if scored is None and self.enable_web_news:
            scored = self._score_from_gdelt(token)
        if scored is None:
            scored = _placeholder_score(token, "news", cycle, bias=0.42, scale=1.5)
        score, confidence, source, payload = scored
        payload["token"] = token
        return EvidenceRecord(
            agent=self.name,
            source=source,
            symbol=symbol,
            kind="news_sentiment",
            observed_at=snapshot.observed_at,
            score=round(score, 6),
            confidence=round(confidence, 6),
            payload=payload,
        )

    def _score_from_rss(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        try:
            headlines = crypto_news_rss(token)
        except Exception as exc:
            LOGGER.debug("crypto rss unavailable token=%s error=%s", token, exc)
            return None
        if not headlines:
            return None
        score = _headline_sentiment(headlines)
        sources = [
            {"title": item["title"], "url": item["url"], "date": item["date"]}
            for item in headlines[:5]
        ]
        return score, clamp(0.55 + abs(score) * 0.25, 0.0, 0.85), "crypto-news-rss", {
            "headline_count": len(headlines),
            "sources": sources,
        }

    def _score_from_skills(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        if self.registry is None:
            return None
        contract = MAJOR_TOKEN_CONTRACTS.get(token)
        if contract is None:
            return None
        # social-hype is a chain-wide leaderboard; find our token's entry on it.
        payload = _run_skill_cached(
            self.registry,
            "crypto-market-rank",
            "social-hype",
            {"chainId": contract["chainId"], "targetLanguage": "en", "timeRange": 1},
        )
        if payload is None:
            return None
        entry = _find_leaderboard_entry(payload, contract)
        if entry is None:
            return None
        hype_info = entry.get("socialHypeInfo", {}) if isinstance(entry, dict) else {}
        sentiment = str(hype_info.get("sentiment", "")).lower()
        score = {"positive": 0.6, "negative": -0.6}.get(sentiment, 0.0)
        return score, clamp(0.6 + abs(score) * 0.25, 0.0, 0.9), "binance-skills-hub", {
            "skill": "crypto-market-rank/social-hype",
            "sentiment": hype_info.get("sentiment"),
            "social_hype": hype_info.get("socialHype"),
            "summary": str(hype_info.get("socialSummaryBrief", ""))[:300],
        }

    def _score_from_web_news(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        query = _TOKEN_NAMES.get(token, token.lower())
        try:
            headlines = run_web_news_search(f"{query} crypto", timelimit="d", max_results=10)
        except Exception as exc:
            LOGGER.debug("web news unavailable token=%s error=%s", token, exc)
            return None
        if not headlines:
            return None
        score = _headline_sentiment(headlines)
        sources = [
            {"title": item["title"], "url": item["url"], "date": item["date"]}
            for item in headlines[:5]
        ]
        return score, clamp(0.5 + abs(score) * 0.25, 0.0, 0.8), "web-news", {
            "headline_count": len(headlines),
            "sources": sources,
        }

    def _score_from_gdelt(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        query = _TOKEN_NAMES.get(token, token.lower())
        headlines = gdelt_headlines(f"{query} crypto")
        if not headlines:
            return None
        score = _headline_sentiment(headlines)
        sources = [
            {"title": item["title"], "url": item["url"], "date": item["date"]}
            for item in headlines[:5]
        ]
        return score, clamp(0.45 + abs(score) * 0.25, 0.0, 0.75), "gdelt-news", {
            "headline_count": len(headlines),
            "sources": sources,
        }


class OnChainFlowAgent:
    """On-chain / smart-money flow, free-first.

    Source order: Binance Skills Hub smart-money signal -> DefiLlama 24h chain
    TVL change -> deterministic placeholder, with provenance recorded in the
    payload.
    """

    name = "onchain_flow_agent"

    def __init__(
        self,
        registry: BinanceSkillRegistry | None = None,
        *,
        enable_defillama: bool = False,
    ) -> None:
        self.registry = registry
        self.enable_defillama = enable_defillama

    def analyze(self, symbol: str, snapshot: MarketSnapshot, cycle: int) -> EvidenceRecord:
        token = symbol.replace("USDT", "")
        # Real major positioning (Binance funding + open interest) first; the Web3
        # smart-money skills only surface meme leaderboards where majors are absent,
        # and DefiLlama chain TVL is a coarse last-resort proxy.
        scored = self._score_from_derivatives(token) if self.enable_defillama else None
        if scored is None:
            scored = self._score_from_skills(token)
        if scored is None and self.enable_defillama:
            scored = self._score_from_defillama(token)
        if scored is None:
            scored = _placeholder_score(token, "onchain", cycle, bias=0.40, scale=1.4)
        score, confidence, source, payload = scored
        payload["token"] = token
        return EvidenceRecord(
            agent=self.name,
            source=source,
            symbol=symbol,
            kind="onchain_flow",
            observed_at=snapshot.observed_at,
            score=round(score, 6),
            confidence=round(confidence, 6),
            payload=payload,
        )

    def _score_from_derivatives(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        try:
            deriv = binance_derivatives(token)
        except Exception as exc:
            LOGGER.debug("derivatives positioning unavailable token=%s error=%s", token, exc)
            return None
        if deriv is None:
            return None
        funding = deriv["funding_rate"]
        # Funding direction & magnitude = net positioning; fade an overcrowded extreme.
        funding_score = clamp(funding / _FUNDING_SATURATION)
        if abs(funding) > _FUNDING_EXTREME:
            funding_score *= 0.4
        oi_change = deriv.get("oi_change_pct")
        oi_score = clamp(oi_change / _OI_CHANGE_SATURATION) if oi_change is not None else 0.0
        # Rising OI amplifies the positioning read; falling OI dampens it.
        score = clamp(funding_score * (1.0 + 0.5 * oi_score))
        read = "net-long" if funding > 0 else "net-short" if funding < 0 else "neutral"
        return score, clamp(0.6 + abs(score) * 0.25, 0.0, 0.85), "binance-derivatives", {
            "funding_rate": funding,
            "open_interest": deriv.get("open_interest"),
            "oi_change_pct": round(oi_change, 6) if oi_change is not None else None,
            "mark_price": deriv.get("mark_price"),
            "positioning": read,
        }

    def _score_from_skills(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        if self.registry is None:
            return None
        contract = MAJOR_TOKEN_CONTRACTS.get(token)
        if contract is None:
            return None
        # smart-money-inflow is a chain-wide net-inflow rank (BSC/Solana only);
        # find our token's entry by contract address.
        payload = _run_skill_cached(
            self.registry,
            "crypto-market-rank",
            "smart-money-inflow",
            {"chainId": contract["chainId"], "period": "24h"},
        )
        if payload is None:
            return None
        entry = _find_leaderboard_entry(payload, contract)
        if entry is None:
            return None
        inflow = _coerce_number(entry.get("inflow")) if isinstance(entry, dict) else None
        if inflow is None:
            return None
        # Squash USD net inflow into [-1, 1]; ~$250k saturates to +-0.5.
        score = clamp(inflow / (abs(inflow) + 250_000.0))
        return score, clamp(0.6 + abs(score) * 0.25, 0.0, 0.9), "binance-skills-hub", {
            "skill": "crypto-market-rank/smart-money-inflow",
            "inflow_usd": inflow,
            "smart_money_traders": entry.get("traders"),
        }

    def _score_from_defillama(self, token: str) -> tuple[float, float, str, dict[str, Any]] | None:
        flow = defillama_tvl_change(token)
        if flow is None:
            return None
        # Coarse flow proxy: a +-5% daily chain-TVL move saturates the score.
        score = clamp(flow["change_pct"] / 0.05)
        return score, clamp(0.5 + abs(score) * 0.2, 0.0, 0.75), "defillama-tvl", {
            "chain": flow["chain"],
            "tvl_now_usd": flow["tvl_now_usd"],
            "tvl_prev_usd": flow["tvl_prev_usd"],
            "tvl_change_pct": round(flow["change_pct"], 6),
        }


def default_agents(
    feed: MarketFeed,
    registry: BinanceSkillRegistry | None = None,
    *,
    enable_web_news: bool = False,
) -> list[SignalAgent]:
    # enable_web_news doubles as the "live free sources allowed" switch: the
    # hermetic/offline path must never reach the network.
    return [
        MarketDataAgent(feed),
        NewsSentimentAgent(registry, enable_web_news=enable_web_news),
        OnChainFlowAgent(registry, enable_defillama=enable_web_news),
    ]


def _run_skill(
    registry: BinanceSkillRegistry,
    skill_name: str,
    command: str,
    params: dict[str, Any],
) -> Any | None:
    try:
        result = registry.run_read_only_cli(skill_name, command, json.dumps(params, sort_keys=True))
    except Exception as exc:
        LOGGER.debug("skill unavailable skill=%s command=%s error=%s", skill_name, command, exc)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _run_skill_cached(
    registry: BinanceSkillRegistry,
    skill_name: str,
    command: str,
    params: dict[str, Any],
) -> Any | None:
    key = f"{skill_name}/{command}:{json.dumps(params, sort_keys=True)}"
    cached = _skill_cache.get(key)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _SKILL_CACHE_TTL_SECONDS:
        return cached[1]
    payload = _run_skill(registry, skill_name, command, params)
    if payload is not None:
        _skill_cache[key] = (now, payload)
    return payload


def _find_leaderboard_entry(payload: Any, contract: dict[str, str]) -> dict[str, Any] | None:
    """Locate our token's row in a chain-wide leaderboard response.

    Matches by contract address first (exact identity), then by the wrapped
    symbol as a fallback for feeds that omit the address.
    """
    address = contract["contractAddress"].lower()
    symbols = {contract["wrappedSymbol"].upper()}
    for entry in _iter_dicts(payload):
        for key in ("contractAddress", "ca"):
            value = entry.get(key)
            if isinstance(value, str) and value.lower() == address:
                return entry
        meta = entry.get("metaInfo")
        if isinstance(meta, dict):
            value = meta.get("contractAddress")
            if isinstance(value, str) and value.lower() == address:
                return entry
            if str(meta.get("symbol", "")).upper() in symbols:
                return entry
        for key in ("symbol", "ticker", "tokenName"):
            if str(entry.get(key, "")).upper() in symbols:
                return entry
    return None


def _iter_dicts(payload: Any, depth: int = 0):
    """Yield candidate row dicts from a nested leaderboard payload."""
    if depth > 4:
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
            yield from _iter_dicts(item, depth + 1)
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_dicts(value, depth + 1)


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _headline_sentiment(headlines: list[dict[str, str]]) -> float:
    positive = 0
    negative = 0
    for item in headlines:
        text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        positive += sum(1 for word in _POSITIVE_WORDS if word in text)
        negative += sum(1 for word in _NEGATIVE_WORDS if word in text)
    total = positive + negative
    if total == 0:
        return 0.0
    return clamp((positive - negative) / total)


def _placeholder_score(
    token: str,
    kind: str,
    cycle: int,
    *,
    bias: float,
    scale: float,
) -> tuple[float, float, str, dict[str, Any]]:
    raw = _stable_unit(token, kind, str(cycle)) - bias
    score = clamp(raw * scale)
    confidence = clamp(0.55 + abs(score) * 0.22, 0.0, 0.85)
    return score, confidence, f"free-first-{kind}-placeholder", {
        "note": "Deterministic placeholder; live skill and web sources were unavailable.",
    }


def _stable_unit(*parts: str) -> float:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
