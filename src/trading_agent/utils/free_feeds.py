"""Free, keyless live data sources used before deterministic placeholders.

Both sources come from the research dossier's free-first shortlist:

- DefiLlama (https://defillama.com/docs/api): daily chain TVL history, used as
  a coarse on-chain flow proxy when Binance Skills Hub is unavailable.
- GDELT DOC 2.0 (https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/):
  last-24h news articles, used when the primary web news search returns
  nothing.

Every fetch is best-effort: any failure returns None and the caller falls
through to the next source. Responses are TTL-cached because TVL is a daily
series and news scoring only needs headline-level freshness.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from trading_agent.core.logging import get_logger

LOGGER = get_logger("free_feeds")

_DEFILLAMA_TVL_URL = "https://api.llama.fi/v2/historicalChainTvl/{chain}"
_GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# Binance public USDS-M futures (keyless): funding rate + open interest are the
# real positioning / crowding signal for the majors (the Web3 smart-money skills
# only surface meme-token chain leaderboards where the majors never appear).
_BINANCE_FAPI = "https://fapi.binance.com"
# Keyless crypto-news RSS feeds (CryptoCompare/CryptoPanic now require keys; these
# do not). Aggregated once and filtered per token.
_CRYPTO_RSS_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
)
_RSS_USER_AGENT = "Mozilla/5.0 (compatible; trading-agent/0.1)"
_TOKEN_NEWS_KEYWORDS = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth "),
    "SOL": ("solana", "sol "),
    "BNB": ("bnb", "binance coin", "binance"),
}
_REQUEST_TIMEOUT_SECONDS = 15
_CACHE_TTL_SECONDS = 30 * 60.0

# Spot token -> DefiLlama chain slug. BTC's chain TVL exists (staking/bridge
# protocols) but is thin; it still beats a synthesized number.
DEFILLAMA_CHAINS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "BNB": "BSC",
}

_cache: dict[str, tuple[float, Any]] = {}


def defillama_tvl_change(token: str) -> dict[str, Any] | None:
    """24h chain-TVL change for the token's primary chain, or None."""
    chain = DEFILLAMA_CHAINS.get(token.upper())
    if chain is None:
        return None
    series = _cached_fetch(_DEFILLAMA_TVL_URL.format(chain=urllib.parse.quote(chain)))
    if not isinstance(series, list) or len(series) < 2:
        return None
    try:
        previous = float(series[-2]["tvl"])
        current = float(series[-1]["tvl"])
    except (KeyError, TypeError, ValueError):
        return None
    if previous <= 0:
        return None
    return {
        "chain": chain,
        "tvl_now_usd": current,
        "tvl_prev_usd": previous,
        "change_pct": (current / previous) - 1.0,
    }


def gdelt_headlines(query: str, *, max_records: int = 15) -> list[dict[str, str]] | None:
    """Last-24h GDELT articles for the query, or None when unavailable/empty."""
    params = urllib.parse.urlencode(
        {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max(1, min(int(max_records), 75)),
            "timespan": "1d",
            "sort": "datedesc",
        }
    )
    payload = _cached_fetch(f"{_GDELT_DOC_URL}?{params}")
    if not isinstance(payload, dict):
        return None
    articles = payload.get("articles")
    if not isinstance(articles, list) or not articles:
        return None
    headlines: list[dict[str, str]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title", "")).strip()
        if not title:
            continue
        headlines.append(
            {
                "title": title,
                "url": str(article.get("url", "")),
                "date": str(article.get("seendate", "")),
                "snippet": "",
            }
        )
    return headlines or None


def binance_derivatives(token: str) -> dict[str, Any] | None:
    """Funding rate + open-interest positioning for a major's perp, or None.

    Keyless Binance futures market data. Funding rate is the crowd's net
    positioning (positive = longs pay shorts = net-long demand; an extreme reading
    is overcrowded/contrarian). Open-interest trend shows whether conviction is
    building (rising) or unwinding (falling). All four majors trade as USDT perps
    under the same symbol as spot (BTCUSDT, ...).
    """
    symbol = f"{token.upper()}USDT"
    premium = _cached_fetch(f"{_BINANCE_FAPI}/fapi/v1/premiumIndex?symbol={symbol}")
    if not isinstance(premium, dict) or "lastFundingRate" not in premium:
        return None
    try:
        funding = float(premium["lastFundingRate"])
        mark = float(premium.get("markPrice", 0) or 0)
    except (TypeError, ValueError):
        return None
    open_interest: float | None = None
    oi_payload = _cached_fetch(f"{_BINANCE_FAPI}/fapi/v1/openInterest?symbol={symbol}")
    if isinstance(oi_payload, dict):
        try:
            open_interest = float(oi_payload["openInterest"])
        except (KeyError, TypeError, ValueError):
            open_interest = None
    oi_change_pct: float | None = None
    history = _cached_fetch(
        f"{_BINANCE_FAPI}/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=7"
    )
    if isinstance(history, list) and len(history) >= 2:
        try:
            first = float(history[0]["sumOpenInterest"])
            last = float(history[-1]["sumOpenInterest"])
            if first > 0:
                oi_change_pct = (last / first) - 1.0
            if open_interest is None:
                open_interest = last
        except (KeyError, TypeError, ValueError):
            oi_change_pct = None
    return {
        "symbol": symbol,
        "funding_rate": funding,
        "mark_price": mark,
        "open_interest": open_interest,
        "oi_change_pct": oi_change_pct,
    }


def crypto_news_rss(token: str, *, max_items: int = 12) -> list[dict[str, str]] | None:
    """Recent crypto-news headlines mentioning the token, from keyless RSS feeds."""
    keywords = _TOKEN_NEWS_KEYWORDS.get(token.upper())
    if not keywords:
        return None
    matched = [
        item
        for item in _cached_rss_items()
        if any(kw in f"{item['title']} {item['snippet']}".lower() for kw in keywords)
    ]
    return matched[:max_items] or None


def _cached_rss_items() -> list[dict[str, str]]:
    """All RSS items across the crypto feeds, TTL-cached as one aggregated list."""
    now = time.monotonic()
    cached = _cache.get("__crypto_rss__")
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    items: list[dict[str, str]] = []
    for url in _CRYPTO_RSS_FEEDS:
        items.extend(_parse_rss(url))
    if items:
        _cache["__crypto_rss__"] = (now, items)
    return items


def _parse_rss(url: str) -> list[dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": _RSS_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
        root = ET.fromstring(body)
    except Exception as exc:  # any feed/parse failure is best-effort
        LOGGER.debug("rss feed unavailable url=%s error=%s", url, exc)
        return []
    source = urllib.parse.urlparse(url).netloc
    items: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        snippet = re.sub(r"<[^>]+>", "", item.findtext("description") or "")[:300].strip()
        items.append(
            {
                "title": title,
                "url": (item.findtext("link") or "").strip(),
                "date": (item.findtext("pubDate") or "").strip(),
                "snippet": snippet,
                "source": source,
            }
        )
    return items


def _cached_fetch(url: str) -> Any | None:
    now = time.monotonic()
    cached = _cache.get(url)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    payload = _fetch_json(url)
    if payload is not None:
        _cache[url] = (now, payload)
    return payload


def _fetch_json(url: str) -> Any | None:
    request = urllib.request.Request(url, headers={"User-Agent": "trading-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except Exception as exc:
        LOGGER.debug("free feed unavailable url=%s error=%s", url, exc)
        return None
