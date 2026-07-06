from __future__ import annotations

import json
import unittest

from trading_agent.utils.binance_skills import (
    COMMAND_USAGE,
    MAJOR_TOKEN_CONTRACTS,
    READ_ONLY_CLI_COMMANDS,
    _normalize_skill_params,
    resolve_major_token,
    run_binance_research_cli,
)


class ResolveMajorTokenTest(unittest.TestCase):
    def test_resolves_spot_symbols_and_bare_tokens(self) -> None:
        for value in ("BTCUSDT", "btc", "BTC", "btcusdt"):
            self.assertEqual(resolve_major_token(value), MAJOR_TOKEN_CONTRACTS["BTC"], value)
        self.assertEqual(resolve_major_token("SOLUSDT"), MAJOR_TOKEN_CONTRACTS["SOL"])
        self.assertIsNone(resolve_major_token("DOGEUSDT"))
        self.assertIsNone(resolve_major_token(123))


class NormalizeSkillParamsTest(unittest.TestCase):
    def test_dynamic_maps_symbol_to_contract(self) -> None:
        # The exact failing call from the 2026-06-12 run logs.
        params, notes = _normalize_skill_params(
            "query-token-info", "dynamic", {"symbol": "BTCUSDT", "currency": "USDT"}
        )
        self.assertEqual(params["chainId"], "56")
        self.assertEqual(params["contractAddress"], MAJOR_TOKEN_CONTRACTS["BTC"]["contractAddress"])
        self.assertNotIn("symbol", params)
        self.assertNotIn("currency", params)
        self.assertTrue(notes)

    def test_kline_interval_shorthand_corrected(self) -> None:
        params, _ = _normalize_skill_params(
            "query-token-info", "kline", {"symbol": "btc", "interval": "15m", "limit": 20}
        )
        self.assertEqual(params["interval"], "15min")
        self.assertEqual(params["limit"], 20)
        self.assertEqual(params["chainId"], "56")

    def test_one_minute_means_minutes_not_months(self) -> None:
        params, _ = _normalize_skill_params(
            "query-token-info", "kline", {"symbol": "eth", "interval": "1m"}
        )
        self.assertEqual(params["interval"], "1min")

    def test_undefined_chain_id_dropped_and_resolved(self) -> None:
        # The exact failing call from the 2026-06-12 run logs.
        params, _ = _normalize_skill_params(
            "trading-signal", "smart-money", {"symbol": "sol", "chainId": "undefined"}
        )
        self.assertEqual(params["chainId"], "CT_501")
        self.assertNotIn("symbol", params)

    def test_social_hype_defaults_injected(self) -> None:
        params, _ = _normalize_skill_params("crypto-market-rank", "social-hype", {"symbol": "bnb"})
        self.assertEqual(params["chainId"], "56")
        self.assertEqual(params["targetLanguage"], "en")
        self.assertEqual(params["timeRange"], 1)

    def test_smart_money_inflow_default_period(self) -> None:
        params, _ = _normalize_skill_params(
            "crypto-market-rank", "smart-money-inflow", {"symbol": "btc"}
        )
        self.assertEqual(params, {"chainId": "56", "period": "24h"})

    def test_search_maps_symbol_to_keyword(self) -> None:
        params, _ = _normalize_skill_params("query-token-info", "search", {"symbol": "solana"})
        self.assertEqual(params, {"keyword": "solana"})

    def test_topic_rush_topic_and_defaults(self) -> None:
        params, _ = _normalize_skill_params("meme-rush", "topic-rush", {"topic": "bitcoin"})
        self.assertEqual(params["keywords"], ["bitcoin"])
        self.assertEqual(params["chainId"], "CT_501")
        self.assertEqual(params["rankType"], 10)
        self.assertEqual(params["sort"], 10)

    def test_explicit_params_pass_through_untouched(self) -> None:
        original = {"chainId": "CT_501", "contractAddress": "So1111", "interval": "1h"}
        params, notes = _normalize_skill_params("query-token-info", "kline", dict(original))
        self.assertEqual(params, original)
        self.assertEqual(notes, [])


class UsageHintTest(unittest.TestCase):
    def test_every_approved_command_has_usage(self) -> None:
        for skill, commands in READ_ONLY_CLI_COMMANDS.items():
            for command in commands:
                self.assertIn(command, COMMAND_USAGE.get(skill, {}), f"{skill}/{command}")

    def test_error_response_includes_usage_and_contracts(self) -> None:
        out = json.loads(run_binance_research_cli("query-token-info", "not-a-command", "{}"))
        self.assertFalse(out["ok"])
        self.assertIn("allowed_commands", out)
        self.assertIn("major_token_contracts", out)


if __name__ == "__main__":
    unittest.main()
