from __future__ import annotations

import unittest

from trading_agent.utils.token_cost import pricing_for, summarize_usage


class TokenCostTest(unittest.TestCase):
    def test_gpt51_cost_math(self) -> None:
        usage = {
            "gpt-5.1": {
                "input_tokens": 100_000,
                "output_tokens": 5_000,
                "input_token_details": {"cache_read": 40_000},
                "output_token_details": {"reasoning": 2_000},
            }
        }
        summary = summarize_usage(usage)
        # uncached 60k*1.25 + cached 40k*0.125 + output 5k*10, all per 1e6
        expected = 60_000 / 1e6 * 1.25 + 40_000 / 1e6 * 0.125 + 5_000 / 1e6 * 10.0
        self.assertAlmostEqual(summary["cost_usd"], round(expected, 6), places=6)
        self.assertEqual(summary["input_tokens"], 100_000)
        self.assertEqual(summary["cached_tokens"], 40_000)
        self.assertEqual(summary["output_tokens"], 5_000)
        self.assertEqual(summary["reasoning_tokens"], 2_000)
        self.assertEqual(summary["total_tokens"], 105_000)
        self.assertTrue(summary["fully_priced"])

    def test_versioned_model_name_resolves_to_base_rate(self) -> None:
        self.assertIsNotNone(pricing_for("gpt-5.1-2026-01-01"))

    def test_unknown_model_counts_tokens_but_flags_unpriced(self) -> None:
        usage = {"some-local-model": {"input_tokens": 1000, "output_tokens": 500}}
        summary = summarize_usage(usage)
        self.assertEqual(summary["total_tokens"], 1500)
        self.assertEqual(summary["cost_usd"], 0.0)
        self.assertFalse(summary["fully_priced"])

    def test_empty_usage_is_zero(self) -> None:
        summary = summarize_usage({})
        self.assertEqual(summary["total_tokens"], 0)
        self.assertEqual(summary["cost_usd"], 0.0)
        self.assertTrue(summary["fully_priced"])


if __name__ == "__main__":
    unittest.main()
