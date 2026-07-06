"""Token accounting and USD cost for LLM usage.

The deep-agent cycle fans out to a supervisor plus six subagents, each running
its own tool-calling loop, so cost is dominated by tokens, not call count. A
LangChain UsageMetadataCallbackHandler (passed via the invoke config) aggregates
usage across every nested model call; ``summarize_usage`` turns that into token
totals and a dollar figure using per-model rates.

Rates are USD per 1,000,000 tokens. ``cached`` is the cache-read input rate
(repeated prompt prefixes are billed ~90% cheaper). Reasoning tokens are already
counted inside ``output_tokens`` by the provider and billed at the output rate.
"""

from __future__ import annotations

from typing import Any

# USD per 1M tokens. Extend as models are added; unknown models are still
# counted in token totals but contribute None to the priced cost.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.1": {"input": 1.25, "cached": 0.125, "output": 10.0},
    # gpt-5.4 (incl. the Azure "gpt-5.4" deployment). PLACEHOLDER rates = gpt-5.1
    # until the authoritative Azure price is confirmed; update if it differs so
    # the promotion gate's "net PnL after LLM cost" is accurate.
    "gpt-5.4": {"input": 1.25, "cached": 0.125, "output": 10.0},
}


def pricing_for(model_name: str) -> dict[str, float] | None:
    """Longest-prefix/substring match so 'gpt-5.1-2026-..' resolves to 'gpt-5.1'."""
    name = (model_name or "").lower()
    best: tuple[str, dict[str, float]] | None = None
    for key, rates in MODEL_PRICING.items():
        if name.startswith(key) or key in name:
            if best is None or len(key) > len(best[0]):
                best = (key, rates)
    return best[1] if best else None


def summarize_usage(usage_by_model: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a UsageMetadataCallbackHandler.usage_metadata mapping.

    Returns token totals, the USD cost of the priced portion, and a flag for
    whether every model that ran had a known rate (so an unknown model is
    visible rather than silently treated as free).
    """
    input_tokens = cached_tokens = output_tokens = reasoning_tokens = 0
    cost = 0.0
    fully_priced = True
    per_model: dict[str, Any] = {}

    for model_name, usage in (usage_by_model or {}).items():
        m_in = int(usage.get("input_tokens", 0) or 0)
        m_out = int(usage.get("output_tokens", 0) or 0)
        m_cached = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
        m_reason = int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)
        input_tokens += m_in
        output_tokens += m_out
        cached_tokens += m_cached
        reasoning_tokens += m_reason

        rates = pricing_for(model_name)
        if rates is None:
            fully_priced = False
            m_cost: float | None = None
        else:
            uncached = max(m_in - m_cached, 0)
            m_cost = (
                uncached / 1e6 * rates["input"]
                + m_cached / 1e6 * rates["cached"]
                + m_out / 1e6 * rates["output"]
            )
            cost += m_cost
        per_model[model_name] = {
            "input_tokens": m_in,
            "cached_tokens": m_cached,
            "output_tokens": m_out,
            "reasoning_tokens": m_reason,
            "cost_usd": None if m_cost is None else round(m_cost, 6),
        }

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 6),
        "fully_priced": fully_priced,
        "per_model": per_model,
    }
