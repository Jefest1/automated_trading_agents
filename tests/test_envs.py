from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from trading_agent.core.config import Settings, load_settings, parse_env_file, resolve_env_reference


class SettingsTest(unittest.TestCase):
    def test_env_test_file_loads_with_redacted_secrets(self) -> None:
        settings = load_settings(".env.test", environ={})
        redacted = settings.redacted()

        self.assertEqual(settings.trading_agent_home, ".trading_agent_test")
        self.assertEqual(settings.trading_agent_mode, "testnet")
        self.assertFalse(settings.enable_llm_supervisor)
        self.assertFalse(redacted["openai_api_key_configured"])
        self.assertFalse(redacted["binance_api_key_configured"])
        self.assertTrue(any("BINANCE_VENUE is unconfirmed" in w for w in redacted["runtime_warnings"]))

    def test_env_file_parser_handles_quotes_and_export_prefix(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.sample"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "export TRADING_AGENT_HOME='custom_home'",
                        'OPENAI_MODEL="gpt-test"',
                    ]
                ),
                encoding="utf-8",
            )

            values = parse_env_file(path)

        self.assertEqual(values["TRADING_AGENT_HOME"], "custom_home")
        self.assertEqual(values["OPENAI_MODEL"], "gpt-test")

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(TRADING_AGENT_MODE="demo")

    def test_azure_openai_envs_auto_select_provider(self) -> None:
        settings = Settings(
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
            AZURE_OPENAI_DEPLOYMENT="trading-agent-gpt",
        )

        self.assertEqual(settings.model_provider, "azure_openai")
        self.assertEqual(settings.resolved_model_name(), "trading-agent-gpt")
        self.assertEqual(settings.model_api_key_env(), "AZURE_OPENAI_API_KEY")
        self.assertEqual(settings.model_api_key_value(), "azure-key")

    def test_explicit_model_provider_overrides_azure_auto_detection(self) -> None:
        settings = Settings(
            MODEL_PROVIDER="openai",
            OPENAI_API_KEY="openai-key",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
            AZURE_OPENAI_DEPLOYMENT="trading-agent-gpt",
        )

        self.assertEqual(settings.model_provider, "openai")
        self.assertEqual(settings.model_api_key_env(), "MODEL_API_KEY" if settings.model_api_key else "OPENAI_API_KEY")
        self.assertEqual(settings.model_api_key_value(), "openai-key")

    def test_blank_model_provider_allows_azure_auto_detection(self) -> None:
        settings = Settings(
            MODEL_PROVIDER="",
            MODEL_API_KEY="generic-openai-compatible-key",
            MODEL_NAME="gpt-5.1",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com/",
            AZURE_OPENAI_DEPLOYMENT="trading-agent-gpt",
        )

        self.assertEqual(settings.model_provider, "azure_openai")
        self.assertEqual(settings.resolved_model_name(), "trading-agent-gpt")
        self.assertEqual(settings.model_api_key_value(), "azure-key")

    def test_env_reference_resolves_only_present_values(self) -> None:
        self.assertEqual(resolve_env_reference("literal", environ={}), "literal")
        self.assertIsNone(resolve_env_reference("env:MISSING", environ={}))
        self.assertEqual(resolve_env_reference("env:PRESENT", environ={"PRESENT": "value"}), "value")


if __name__ == "__main__":
    unittest.main()
