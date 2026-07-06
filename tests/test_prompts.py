from __future__ import annotations

import unittest

from trading_agent.prompts import PROMPTS


class PromptRegistryTest(unittest.TestCase):
    def test_all_required_agent_prompts_are_versioned_and_safe(self) -> None:
        required = {
            "supervisor",
            "market_research",
            "technical_analyst",
            "news_research",
            "onchain_research",
            # Advisory debate agents (FULL cycles only); not in REQUIRED_AGENTS.
            "bull_researcher",
            "bear_researcher",
            "strategy",
            "risk_review",
            "reporting",
        }

        self.assertEqual(set(PROMPTS), required)
        for name, prompt in PROMPTS.items():
            self.assertEqual(prompt.name, name)
            self.assertTrue(prompt.version)
            self.assertIn("no direct trade execution", prompt.text.lower())


if __name__ == "__main__":
    unittest.main()
