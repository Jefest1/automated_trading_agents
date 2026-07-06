from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from ddgs import DDGS
from langchain_core.tools import tool

from trading_agent.core.logging import get_logger
from trading_agent.utils.free_feeds import crypto_news_rss, gdelt_headlines

LOGGER = get_logger("web_search")

_FETCH_TIMEOUT_SECONDS = 10
_FETCH_USER_AGENT = "trading-agent/0.1 (research; read-only)"
_SKIP_TAGS = {"script", "style", "noscript", "head", "title", "svg"}

# Jina is keyless for moderate use (no subscription); an optional JINA_API_KEY
# in the environment only raises the rate limit. It replaced Helium (paywalled)
# and shores up DuckDuckGo, which has been returning empty result sets.
_JINA_SEARCH_URL = "https://s.jina.ai/"
_JINA_READER_URL = "https://r.jina.ai/"


def _jina_headers(accept: str) -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": _FETCH_USER_AGENT}
    key = os.environ.get("JINA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _jina_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Keyless Jina search (s.jina.ai) -> [{title, url, snippet, date}]."""
    request = urllib.request.Request(
        _JINA_SEARCH_URL + urllib.parse.quote(query),
        headers=_jina_headers("application/json"),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))
    rows: list[dict[str, str]] = []
    for item in (payload.get("data") or [])[:max_results]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("description") or item.get("content") or "")[:500],
                "date": str(item.get("date", "")),
            }
        )
    return rows


# Markdown image embeds (![alt](https://cdn...)) and bare image/asset URLs are
# pure tokens with no analytical value; the news dumps were ~30% image CDN links.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BARE_IMG_URL = re.compile(r"https?://\S+\.(?:png|jpe?g|gif|webp|svg)\S*", re.IGNORECASE)


def _strip_page_noise(text: str) -> str:
    text = _MD_IMAGE.sub("", text)
    text = _BARE_IMG_URL.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _jina_reader(url: str, max_chars: int) -> str:
    """Keyless Jina Reader (r.jina.ai) -> clean markdown for a page (handles JS)."""
    request = urllib.request.Request(
        _JINA_READER_URL + url, headers=_jina_headers("text/plain"), method="GET"
    )
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
        body = response.read(2_000_000).decode("utf-8", errors="replace")
    return _strip_page_noise(body)[:max_chars]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _ddg_text(query: str, max_results: int) -> list[dict[str, str]]:
    with DDGS() as client:
        rows = client.text(query, max_results=max_results)
    return [
        {"title": row.get("title", ""), "url": row.get("href", row.get("url", "")), "snippet": row.get("body", "")}
        for row in rows or []
    ]


def run_web_search(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Keyless web search: Jina (s.jina.ai) first, DuckDuckGo fallback.

    Returns [{title, url, snippet}]. Each backend is guarded; an empty/blocked
    source falls through to the next instead of raising."""
    try:
        rows = _jina_search(query, max_results)
        if rows:
            return [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]} for r in rows]
    except Exception as exc:
        LOGGER.info("jina search unavailable query=%s detail=%s; trying DuckDuckGo", query, exc)
    try:
        return _ddg_text(query, max_results)
    except Exception as exc:
        LOGGER.info("duckduckgo text search empty/failed query=%s detail=%s", query, exc)
        return []


def run_web_news_search(query: str, timelimit: str = "d", max_results: int = 8) -> list[dict[str, str]]:
    """Recent-news search; returns [{title, url, source, date, snippet}].

    Source order: DuckDuckGo news (dated) -> Jina search (broad, keyless) -> the
    GDELT fallback is applied by the tool wrapper. timelimit d/w/m (DDG only)."""
    try:
        with DDGS() as client:
            rows = client.news(query, timelimit=timelimit, max_results=max_results)
        results = [
            {
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "source": row.get("source", ""),
                "date": row.get("date", ""),
                "snippet": row.get("body", ""),
            }
            for row in rows or []
        ]
        if results:
            return results
    except Exception as exc:
        LOGGER.info("duckduckgo news empty/failed query=%s detail=%s; trying Jina", query, exc)
    try:
        return [
            {"title": r["title"], "url": r["url"], "source": "", "date": r["date"], "snippet": r["snippet"]}
            for r in _jina_search(f"{query} latest news", max_results)
        ]
    except Exception as exc:
        LOGGER.info("jina news unavailable query=%s detail=%s", query, exc)
        return []


def run_fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a public URL as readable text: Jina Reader (r.jina.ai) first for
    clean JS-rendered markdown, then a local HTML-strip fallback."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    try:
        text = _jina_reader(url, max_chars)
        if text:
            return text
    except Exception as exc:
        LOGGER.info("jina reader unavailable url=%s detail=%s; falling back to raw fetch", url, exc)
    request = urllib.request.Request(url, headers={"User-Agent": _FETCH_USER_AGENT}, method="GET")
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
        body = response.read(1_000_000).decode("utf-8", errors="replace")
    extractor = _TextExtractor()
    extractor.feed(body)
    return _strip_page_noise(extractor.text())[:max_chars]


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """Search the live web (free DuckDuckGo metasearch).

    Use for market context, macro events, project announcements, and anything
    not covered by Binance research skills. Returns a JSON array of
    {title, url, snippet}. Always cite the url of any result you rely on.
    """
    try:
        results = run_web_search(query, max_results=max_results)
    except Exception as exc:
        LOGGER.warning("web_search failed query=%s error=%s", query, exc)
        return json.dumps({"ok": False, "error": str(exc), "query": query}, sort_keys=True)
    return json.dumps({"ok": True, "query": query, "results": results}, sort_keys=True)


@tool
def web_news_search(query: str, timelimit: str = "d", max_results: int = 8) -> str:
    """Search recent news articles (free DuckDuckGo news, GDELT fallback).

    timelimit: "d" = last day, "w" = last week, "m" = last month.
    Returns a JSON array of {title, url, source, date, snippet}. Always cite
    the url and date of any headline you rely on as evidence. An empty result
    set is a normal outcome (ok=true, results=[]), not an error.
    """
    results: list[dict[str, str]] = []
    source = "duckduckgo"
    try:
        results = run_web_news_search(query, timelimit=timelimit, max_results=max_results)
    except Exception as exc:
        # DDGS raises "No results found." on an empty day; that is not a failure.
        LOGGER.info("web_news_search empty/failed query=%s detail=%s; trying GDELT", query, exc)
    if not results:
        fallback = gdelt_headlines(query, max_records=max_results)
        if fallback:
            results = fallback
            source = "gdelt"
    return json.dumps(
        {"ok": True, "query": query, "source": source, "results": results}, sort_keys=True
    )


@tool
def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a public web page and return its readable text (HTML stripped).

    Use after web_search/web_news_search to read a source before citing it.
    Image/asset links are stripped and output is truncated to max_chars
    characters (~6k is enough for a news article's substance).
    """
    try:
        text = run_fetch_url(url, max_chars=max_chars)
    except Exception as exc:
        LOGGER.warning("fetch_url failed url=%s error=%s", url, exc)
        return json.dumps({"ok": False, "error": str(exc), "url": url}, sort_keys=True)
    return json.dumps({"ok": True, "url": url, "text": text}, sort_keys=True)


@tool
def get_crypto_news(symbol: str, max_items: int = 10) -> str:
    """Recent crypto-news headlines for a major, from keyless RSS feeds.

    Reliable primary news source for BTC/ETH/SOL/BNB (CoinDesk, Cointelegraph,
    Decrypt), filtered to the token. Use this BEFORE the generic web_news_search,
    which is flaky/rate-limited. Returns {ok, symbol, headlines:[{title,url,date,
    source,snippet}]}. An empty list is a valid "no fresh headlines" finding.
    """
    token = symbol.upper().removesuffix("USDT")
    try:
        headlines = crypto_news_rss(token, max_items=max(1, min(int(max_items), 25)))
    except Exception as exc:
        LOGGER.warning("get_crypto_news failed symbol=%s error=%s", symbol, exc)
        return json.dumps({"ok": False, "error": str(exc), "symbol": symbol}, sort_keys=True)
    return json.dumps(
        {"ok": True, "symbol": symbol.upper(), "headlines": headlines or []},
        sort_keys=True,
        ensure_ascii=False,
    )


WEB_RESEARCH_TOOLS = [get_crypto_news, web_search, web_news_search, fetch_url]
