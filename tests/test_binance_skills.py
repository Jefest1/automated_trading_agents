from __future__ import annotations

import unittest

import json

from trading_agent.utils.binance_skills import BinanceSkillRegistry, run_binance_research_cli


class BinanceSkillsTest(unittest.TestCase):
    def test_installed_binance_skills_are_discovered_and_classified(self) -> None:
        registry = BinanceSkillRegistry()
        skills = {skill.name: skill for skill in registry.list_skills()}

        self.assertIn("binance", skills)
        self.assertIn("query-token-info", skills)
        self.assertEqual(skills["binance"].category, "execution_or_auth_sensitive")
        self.assertEqual(skills["query-token-info"].category, "read_only_cli")
        self.assertIn("kline", skills["query-token-info"].commands)

    def test_readonly_skill_source_is_available_for_deep_agents(self) -> None:
        sources = BinanceSkillRegistry().read_only_skill_source_paths()

        self.assertIn("skills/binance-readonly", sources)

    def test_execution_sensitive_skills_cannot_run_through_readonly_cli_wrapper(self) -> None:
        registry = BinanceSkillRegistry()

        with self.assertRaises(ValueError):
            registry.run_read_only_cli("binance", "spot", "{}")

    def test_unapproved_command_is_rejected(self) -> None:
        registry = BinanceSkillRegistry()

        with self.assertRaises(ValueError):
            registry.run_read_only_cli("query-token-info", "new-order", "{}")

    def test_llm_research_tool_returns_structured_error_for_bad_command(self) -> None:
        payload = json.loads(run_binance_research_cli("trading-signal", "dummy", "{}"))

        self.assertFalse(payload["ok"])
        self.assertIn("not allowed", payload["error"])
        self.assertIn("trading-signal", payload["allowed_commands"])


if __name__ == "__main__":
    unittest.main()
