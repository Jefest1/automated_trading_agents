from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


DEFAULT_HOME = Path(".trading_agent")


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    app_env: str = Field(default="test", alias="APP_ENV")
    trading_agent_home: str = Field(default=".trading_agent", alias="TRADING_AGENT_HOME")
    trading_agent_mode: str = Field(default="testnet", alias="TRADING_AGENT_MODE")
    trading_agent_execution_mode: str = Field(default="testnet", alias="TRADING_AGENT_EXECUTION_MODE")
    enable_testnet_orders: bool = Field(default=False, alias="TRADING_AGENT_ENABLE_TESTNET_ORDERS")
    enable_live_orders: bool = Field(default=False, alias="TRADING_AGENT_ENABLE_LIVE_ORDERS")
    subagent_models_json: str | None = Field(default=None, alias="TRADING_AGENT_SUBAGENT_MODELS")
    enable_llm_supervisor: bool = Field(default=False, alias="TRADING_AGENT_ENABLE_LLM_SUPERVISOR")
    live_data: bool = Field(default=False, alias="TRADING_AGENT_LIVE_DATA")

    model_provider: str | None = Field(default=None, alias="MODEL_PROVIDER")
    model_name: str | None = Field(default=None, alias="MODEL_NAME")
    model_base_url: str | None = Field(default=None, alias="MODEL_BASE_URL")
    model_api_key: SecretStr | None = Field(default=None, alias="MODEL_API_KEY")

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-5.1", alias="OPENAI_MODEL")

    azure_openai_api_key: SecretStr | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_project_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_PROJECT_ENDPOINT")
    azure_openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str | None = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")

    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-2.5-flash", alias="GOOGLE_MODEL")

    # Optional: raises Jina web-search/reader rate limits. Keyless works without it.
    jina_api_key: SecretStr | None = Field(default=None, alias="JINA_API_KEY")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    binance_venue: str = Field(default="unconfirmed", alias="BINANCE_VENUE")
    binance_api_key: SecretStr | None = Field(default=None, alias="BINANCE_API_KEY")
    binance_api_secret: SecretStr | None = Field(default=None, alias="BINANCE_API_SECRET")
    binance_api_base_url: str = Field(default="https://api.binance.com", alias="BINANCE_API_BASE_URL")
    binance_us_api_base_url: str = Field(default="https://api.binance.us", alias="BINANCE_US_API_BASE_URL")

    nemoclaw_enabled: bool = Field(default=False, alias="NEMOCLAW_ENABLED")
    nemoclaw_sandbox: str = Field(default="trading-agent", alias="NEMOCLAW_SANDBOX")
    nemoclaw_policy_profile: str = Field(
        default="trading-agent-readonly-research",
        alias="NEMOCLAW_POLICY_PROFILE",
    )
    nemoclaw_policy_explain_path: str = Field(
        default=".trading_agent/nemoclaw/POLICY.md",
        alias="NEMOCLAW_POLICY_EXPLAIN_PATH",
    )
    nemoclaw_shell_enabled: bool = Field(default=False, alias="NEMOCLAW_SHELL_ENABLED")

    log_level: str = Field(default="INFO", alias="TRADING_AGENT_LOG_LEVEL")
    log_to_stderr: bool = Field(default=True, alias="TRADING_AGENT_LOG_TO_STDERR")
    log_to_file: bool = Field(default=True, alias="TRADING_AGENT_LOG_TO_FILE")

    @field_validator("trading_agent_mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"testnet", "live"}:
            raise ValueError("TRADING_AGENT_MODE must be testnet or live (paper trading removed)")
        return lowered

    @field_validator("trading_agent_execution_mode")
    @classmethod
    def validate_execution_mode(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"testnet", "live"}:
            raise ValueError("TRADING_AGENT_EXECUTION_MODE must be testnet or live (paper trading removed)")
        return lowered

    @field_validator("binance_venue")
    @classmethod
    def validate_binance_venue(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"unconfirmed", "binance.com", "binance.us", "testnet"}:
            raise ValueError("BINANCE_VENUE must be unconfirmed, binance.com, binance.us, or testnet")
        return lowered

    @field_validator(
        "enable_llm_supervisor",
        "enable_testnet_orders",
        "enable_live_orders",
        "live_data",
        "nemoclaw_enabled",
        "nemoclaw_shell_enabled",
        "log_to_stderr",
        "log_to_file",
        mode="before",
    )
    @classmethod
    def parse_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or value == "":
            return False
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError(f"invalid boolean env value: {value!r}")

    @field_validator("model_provider", mode="before")
    @classmethod
    def blank_model_provider_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_provider")
    @classmethod
    def validate_model_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if lowered not in {"openai", "azure_openai", "anthropic", "google_genai", "ollama", "openrouter"}:
            raise ValueError(
                "MODEL_PROVIDER must be openai, azure_openai, anthropic, google_genai, ollama, or openrouter"
            )
        return lowered

    @field_validator(
        "model_api_key",
        "openai_api_key",
        "azure_openai_api_key",
        "anthropic_api_key",
        "google_api_key",
        "jina_api_key",
        "binance_api_key",
        "binance_api_secret",
        mode="before",
    )
    @classmethod
    def blank_secret_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "model_name",
        "model_base_url",
        "azure_openai_endpoint",
        "azure_openai_project_endpoint",
        "azure_openai_deployment",
        "azure_openai_api_version",
        mode="before",
    )
    @classmethod
    def blank_string_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("TRADING_AGENT_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return upper

    @model_validator(mode="after")
    def infer_model_provider(self) -> "Settings":
        if self.model_provider is None:
            object.__setattr__(self, "model_provider", self.detect_model_provider())
        return self

    def detect_model_provider(self) -> str:
        """Infer the model provider from provider-specific env vars."""
        if any(
            [
                self.azure_openai_api_key is not None,
                bool(self.azure_openai_endpoint),
                bool(self.azure_openai_project_endpoint),
                bool(self.azure_openai_deployment),
            ]
        ):
            return "azure_openai"
        if self.anthropic_api_key is not None:
            return "anthropic"
        if self.google_api_key is not None:
            return "google_genai"
        if self.openai_api_key is not None or self.model_api_key is not None:
            return "openai"
        return "openai"

    def resolved_model_name(self) -> str:
        """Model name for the active provider, honoring MODEL_NAME override."""
        if self.model_provider == "azure_openai":
            return self.azure_openai_deployment or self.model_name or ""
        if self.model_name:
            return self.model_name
        if self.model_provider == "anthropic":
            return self.anthropic_model
        if self.model_provider == "google_genai":
            return self.google_model
        if self.model_provider == "ollama":
            return self.ollama_model
        return self.openai_model

    def model_api_key_env(self) -> str:
        """Env var name expected to carry the active provider's API key."""
        if self.model_provider == "azure_openai" and self.azure_openai_api_key is not None:
            return "AZURE_OPENAI_API_KEY"
        if self.model_api_key is not None:
            return "MODEL_API_KEY"
        return {
            "openai": "OPENAI_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
        }.get(self.model_provider, "MODEL_API_KEY")

    def model_api_key_value(self) -> str | None:
        """API key for the active provider, with Azure using its provider-specific key first."""
        if self.model_provider == "azure_openai" and self.azure_openai_api_key is not None:
            return self.azure_openai_api_key.get_secret_value()
        if self.model_api_key is not None:
            return self.model_api_key.get_secret_value()
        provider_key = {
            "openai": self.openai_api_key,
            "azure_openai": self.azure_openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google_genai": self.google_api_key,
        }.get(self.model_provider)
        if provider_key is not None:
            return provider_key.get_secret_value()
        return None

    def model_requires_api_key(self) -> bool:
        return self.model_provider not in {"ollama"}

    def exchange_base_url(self) -> str:
        """Order-execution base URL for the active venue/mode.

        Testnet always pins to testnet.binance.vision; live resolves
        binance.com / binance.us. Market DATA stays on the public production
        API regardless (see utils/market_data.py) because testnet prices
        diverge from the real market.
        """
        if self.trading_agent_execution_mode == "testnet" or self.binance_venue == "testnet":
            return "https://testnet.binance.vision/api"
        base = self.binance_us_api_base_url if self.binance_venue == "binance.us" else self.binance_api_base_url
        base = base.rstrip("/")
        return base if base.endswith("/api") else f"{base}/api"

    def live_order_blockers(self) -> list[str]:
        """Every condition that still blocks live order submission (empty = go)."""
        blockers: list[str] = []
        if self.trading_agent_execution_mode != "live":
            blockers.append("TRADING_AGENT_EXECUTION_MODE must be live")
        if not self.enable_live_orders:
            blockers.append("TRADING_AGENT_ENABLE_LIVE_ORDERS=true is required")
        if self.binance_venue not in {"binance.com", "binance.us"}:
            blockers.append("BINANCE_VENUE must be binance.com or binance.us")
        if self.binance_api_key is None or self.binance_api_secret is None:
            blockers.append("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        return blockers

    def runtime_warnings(self) -> list[str]:
        warnings: list[str] = []
        if (
            self.enable_llm_supervisor
            and self.model_requires_api_key()
            and self.model_api_key_value() is None
        ):
            warnings.append(
                f"TRADING_AGENT_ENABLE_LLM_SUPERVISOR is true but {self.model_api_key_env()} is not set"
            )
        if self.enable_llm_supervisor and self.model_provider == "azure_openai":
            if not self.azure_openai_endpoint:
                warnings.append(
                    "TRADING_AGENT_ENABLE_LLM_SUPERVISOR is true but AZURE_OPENAI_ENDPOINT is not set"
                )
            if not self.resolved_model_name():
                warnings.append(
                    "TRADING_AGENT_ENABLE_LLM_SUPERVISOR is true but AZURE_OPENAI_DEPLOYMENT or MODEL_NAME is not set"
                )
            if not self.azure_openai_api_version:
                warnings.append(
                    "TRADING_AGENT_ENABLE_LLM_SUPERVISOR is true but AZURE_OPENAI_API_VERSION is not set"
                )
        if self.trading_agent_execution_mode == "live":
            for blocker in self.live_order_blockers():
                warnings.append(f"Live execution blocked: {blocker}")
            warnings.append(
                "Live execution also requires config.json live.enabled=true and live.venue_confirmed=true"
            )
        if self.trading_agent_execution_mode == "testnet":
            if not self.enable_testnet_orders:
                warnings.append(
                    "TRADING_AGENT_EXECUTION_MODE=testnet requires TRADING_AGENT_ENABLE_TESTNET_ORDERS=true"
                )
            if self.binance_venue != "testnet":
                warnings.append("Testnet execution requires BINANCE_VENUE=testnet")
            if "testnet.binance.vision" not in self.binance_api_base_url:
                warnings.append("Testnet execution requires BINANCE_API_BASE_URL=https://testnet.binance.vision/api")
            if self.binance_api_key is None or self.binance_api_secret is None:
                warnings.append("Testnet execution requires BINANCE_API_KEY and BINANCE_API_SECRET")
        if self.binance_venue == "unconfirmed":
            warnings.append("BINANCE_VENUE is unconfirmed; live exchange behavior must remain disabled")
        if self.nemoclaw_shell_enabled and not self.nemoclaw_enabled:
            warnings.append("NEMOCLAW_SHELL_ENABLED is true but NEMOCLAW_ENABLED is false")
        return warnings

    def redacted(self) -> dict[str, Any]:
        return {
            "app_env": self.app_env,
            "trading_agent_home": self.trading_agent_home,
            "trading_agent_mode": self.trading_agent_mode,
            "trading_agent_execution_mode": self.trading_agent_execution_mode,
            "enable_testnet_orders": self.enable_testnet_orders,
            "enable_live_orders": self.enable_live_orders,
            "enable_llm_supervisor": self.enable_llm_supervisor,
            "live_data": self.live_data,
            "model_provider": self.model_provider,
            "model_name": self.resolved_model_name(),
            "model_base_url": self.model_base_url,
            "model_api_key_configured": self.model_api_key_value() is not None,
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "openai_api_key_configured": self.openai_api_key is not None,
            "azure_openai_endpoint": self.azure_openai_endpoint,
            "azure_openai_project_endpoint": self.azure_openai_project_endpoint,
            "azure_openai_deployment": self.azure_openai_deployment,
            "azure_openai_api_version": self.azure_openai_api_version,
            "azure_openai_api_key_configured": self.azure_openai_api_key is not None,
            "anthropic_api_key_configured": self.anthropic_api_key is not None,
            "binance_venue": self.binance_venue,
            "exchange_base_url": self.exchange_base_url(),
            "binance_api_base_url": self.binance_api_base_url,
            "binance_us_api_base_url": self.binance_us_api_base_url,
            "binance_api_key_configured": self.binance_api_key is not None,
            "binance_api_secret_configured": self.binance_api_secret is not None,
            "nemoclaw_enabled": self.nemoclaw_enabled,
            "nemoclaw_sandbox": self.nemoclaw_sandbox,
            "nemoclaw_policy_profile": self.nemoclaw_policy_profile,
            "nemoclaw_policy_explain_path": self.nemoclaw_policy_explain_path,
            "nemoclaw_shell_enabled": self.nemoclaw_shell_enabled,
            "log_level": self.log_level,
            "log_to_stderr": self.log_to_stderr,
            "log_to_file": self.log_to_file,
            "runtime_warnings": self.runtime_warnings(),
        }


@dataclass(slots=True)
class RiskConfig:
    allowed_symbols: list[str] = field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    )
    max_open_positions: int = 3
    third_order_min_confidence: float = 0.90
    # Conviction floor. This is now a NOISE floor, not a participation gate:
    # conviction-scaled sizing (SizingConfig) takes a small position for a
    # marginal-but-valid setup instead of vetoing it, so this only rejects
    # genuinely low-conviction noise. Matches SizingConfig.conviction_floor.
    # Testnet uses an even looser floor so paper trading generates fills + data
    # (see testnet_min_confidence and RiskGovernor.evaluate).
    min_confidence: float = 0.50
    testnet_min_confidence: float = 0.40
    # Evidence freshness: this is an HOURLY swing desk entering off 1h/4h
    # structure, so a read stays usable for ~an hour (a 20-min-old observation is
    # not a different market on this horizon).
    stale_data_seconds: int = 60 * 60
    per_trade_risk_fraction: float = 0.01
    # Swing-width bracket: stops/targets are sized for hours-to-days holds off
    # 1h/4h structure, not an intraday scalp. Feeds the deterministic StrategyAgent
    # baseline and the legacy single-leg bracket (when exits.enabled is False).
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.06
    # Correlated-exposure cap: BTC/ETH/SOL/BNB move together, so several majors
    # long at once is ONE leveraged beta bet. This caps total open notional across
    # the correlated majors (a portfolio limit the position-count cap alone misses).
    # 0 disables. Default = 2x the single-position book unit.
    max_correlated_notional_usd: float = 200.0
    # Daily-loss circuit breaker: once realized PnL for the UTC day drops to
    # -daily_loss_halt_pct * capital_budget, new BUYs are halted (HOLD/CLOSE still
    # allowed) and the rejection is recorded as a real risk-control breach. 0 disables.
    daily_loss_halt_pct: float = 0.05
    # Must exceed the modeled round-trip cost (backtest model: 2x fee_bps + slippage_bps
    # = 25 bps), otherwise every minimum-edge trade loses money by construction.
    min_expected_edge_bps: float = 30.0
    # Per-trade risk sizing assumes the stop does not fill exactly at its price:
    # worst-case loss = notional * (stop_loss_pct + assumed_slippage + gap buffer).
    # A limit stop on spot can gap through in a fast move, so the buffer is the
    # extra cushion operators want priced into the per-trade risk check.
    assumed_slippage_bps: float = 5.0
    stop_gap_buffer_pct: float = 0.0
    # Auditability: a supervisor BUY must cite evidence ids that resolve to
    # records actually gathered this cycle. When true, a BUY whose evidence_refs
    # resolve to nothing is rejected instead of silently re-scored on the
    # symbol's evidence (which lets the LLM fabricate its consultation trail).
    require_evidence_refs: bool = True
    # Timeframe whose ATR sizes maker-pullback entry offsets and the runner trail.
    # A swing desk sizes off the 1h candle, not 15m (see maker_pullback_price and
    # ExitConfig.trail_atr_mult; both read snapshot.atr, set from this interval).
    atr_interval: str = "1h"
    # Maker-pullback entries: a BUY limit rests BELOW the current bid so it fills
    # on a normal dip as a maker order (lower fees, better entry), not by chasing
    # the ask. Offset = entry_atr_mult * ATR(atr_interval), clamped to a band so it
    # neither sits on the bid nor sits so far it never fills. The band is wider for
    # swing so a deeper 1h pullback entry is still reachable.
    entry_atr_mult: float = 0.3
    entry_min_offset_bps: float = 5.0
    entry_max_offset_pct: float = 0.025
    # A BUY whose limit is above the bid by more than this many bps is rejected
    # ("entry must rest below the market, not chase"). Small tolerance absorbs the
    # research->gate drift; 0 would reject any limit at/above bid.
    max_cross_spread_bps: float = 5.0
    # Anti-churn cooldown: a freshly opened long cannot be CLOSED discretionarily by
    # the agent team for this many hours (matches the bid TTL / hourly-swing horizon),
    # so the desk holds for hours instead of re-litigating a still-valid position every
    # cycle. The deterministic stop-loss/exit-ladder still fires anytime, and an
    # operator close bypasses this — it only gates the LLM's discretionary exits.
    min_hold_hours: float = 4.0
    # Hard maker discipline: when true, the deterministic gate OVERRIDES the LLM's
    # proposed BUY limit with the maker-pullback price (bid - entry_atr_mult*ATR,
    # clamped), so an entry can never chase/cross the spread. The LLM's suggested
    # price is recorded for audit but never sent.
    hard_maker_entry: bool = True
    # --- Demand-zone laddered entries -------------------------------------- #
    # Instead of defaulting to WAIT when a liked name is poorly located, the desk
    # rests scaled limit bids INTO the next computed demand zone (well below price)
    # with the stop below the zone. These knobs govern that behaviour.
    #
    # Allow a zone-anchored bid to sit this far below market (the old flat 2% price
    # leash forbade real swing supports, which sit 3-6% down). Only bids tied to a
    # confirmed support zone get this depth; un-anchored bids keep the tight leash.
    max_bid_depth_pct: float = 0.08
    # A bid NOT anchored to a demand zone may still only sit this far from market
    # (the legacy shallow-pullback behaviour, preserved as the fallback).
    unanchored_max_deviation_pct: float = 0.02
    # When true, a deep bid (beyond unanchored_max_deviation_pct) is allowed only if
    # it rests inside/just above the low of a computed support zone. False reverts to
    # the old flat-deviation check for everyone.
    require_zone_anchored_bids: bool = True
    # Ladder: split the conviction notional across this many resting bids at
    # successive support zones (nearest first). 1 disables laddering.
    ladder_legs: int = 3
    # Notional split across ladder legs (nearest..deepest). Normalised at use; extra
    # legs beyond the list reuse the last weight. Front-loaded so the most likely
    # fill carries the most size.
    ladder_size_split: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    # A resting bid is cancelled if unfilled after this many minutes (GTD-style), so
    # a stale bid never fills days later into a dead thesis. 0 disables expiry.
    bid_ttl_minutes: int = 240
    # Minimum reward:risk a zone-anchored entry must clear (target vs zone stop). A
    # swing setup that cannot show this from the bid is not worth the capital.
    min_reward_risk: float = 1.5
    # Stop sits this fraction of the zone height BELOW the zone low (a clean break of
    # the demand zone invalidates the thesis -> out small).
    zone_stop_buffer_pct: float = 0.005
    # In a confirmed downtrend the desk does not bid every zone; it only works the
    # single deepest, highest-confluence zone and at reduced size (this multiplier).
    downtrend_size_mult: float = 0.5
    # Live mark-to-market cache TTL: open-position unrealized PnL is refreshed from
    # the public spot API at most this often (operator commands + loop heartbeat).
    mark_refresh_seconds: int = 300
    # How often the fast bracket monitor evaluates open-position TP/SL exits between
    # (hourly) decision cycles. Decoupled from the LLM cadence so a stop/trailing
    # touch is acted on within ~this many seconds instead of at the next cycle.
    bracket_monitor_seconds: int = 60


@dataclass(slots=True)
class BacktestConfig:
    """Fee/slippage and base order size for OFFLINE historical simulation
    (`backtest` and `backtest-decisions`). This is not a live trading mode —
    the system trades real spot on testnet/live only."""

    order_notional_usd: float = 100.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0


@dataclass(slots=True)
class StrategyConfig:
    """Tunable constants for the deterministic StrategyAgent baseline.

    These were previously hard-coded magic numbers; exposing them lets the
    edge transfer function and confidence model be tuned and backtested
    instead of taken on faith. ``edge_scale_bps`` maps a combined evidence
    score in [-1, 1] onto an expected edge; the confidence model is
    confidence_base + |score|*score_coef + avg_agent_confidence*agreement_coef.
    """

    agent_weights: dict[str, float] = field(
        default_factory=lambda: {
            "market_data_agent": 0.50,
            "news_sentiment_agent": 0.25,
            "onchain_flow_agent": 0.25,
        }
    )
    default_agent_weight: float = 0.10
    edge_scale_bps: float = 120.0
    confidence_base: float = 0.50
    confidence_score_coef: float = 0.45
    confidence_agreement_coef: float = 0.18
    confidence_cap: float = 0.99


@dataclass(slots=True)
class SizingConfig:
    """Conviction-scaled position sizing.

    Replaces the old all-or-nothing model (fixed notional gated behind a hard
    confidence floor, which froze the desk in choppy tapes). Target notional now
    scales with conviction x edge, de-risked by a fractional-Kelly multiplier,
    current volatility, the macro regime, and data quality. A marginal-but-valid
    setup takes a SMALL real position instead of being vetoed, so the desk keeps
    participating (and learning) instead of waiting for near-certainty.

    The book unit is ``LiveConfig.capital_budget_usd``; this sizer scales it
    between ``min_size_frac`` and ``kelly_fraction`` of that unit. The deterministic
    RiskGovernor still independently validates per-trade risk and exposure, so the
    sizer only ever PROPOSES a size the gate can still shrink or reject.
    """

    # Conviction below conviction_floor never reaches the sizer (RiskGovernor
    # min_confidence rejects it as noise). At/above conviction_full the conviction
    # factor saturates at 1.0; between them it ramps linearly.
    conviction_floor: float = 0.50
    conviction_full: float = 0.80
    # expected_edge_bps at/above this saturates the edge factor at 1.0.
    edge_full_bps: float = 60.0
    # Smallest fraction of the book unit a just-valid setup takes (so a marginal
    # trade is small, not skipped). At full strength the fraction reaches
    # kelly_fraction (<=1.0; lower = global de-risk, classic fractional Kelly).
    min_size_frac: float = 0.15
    kelly_fraction: float = 1.0
    # Volatility targeting: when ATR%% (on RiskConfig.atr_interval, i.e. 1h) exceeds
    # this, notional is trimmed by vol_target_pct / atr_pct so unusually volatile
    # names get smaller size. Set against the 1h ATR basis (~2x the old 15m basis).
    vol_target_pct: float = 0.02
    # Multiplier callers apply in a risk-off / strong-USD regime (see WP2).
    regime_risk_off_mult: float = 0.5
    # Exchange min-notional floor; a computed size below this returns 0 (skip).
    min_notional_usd: float = 15.0
    # Hard cap per position (also clamped to the available book unit).
    max_notional_usd: float = 100.0


@dataclass(slots=True)
class ExitConfig:
    """Tiered take-profit scale-out with a ratcheting/trailing stop.

    Instead of a single all-or-nothing take-profit, an open position is exited
    in tiers: each ``take_profit_tiers`` entry sells ``size_pct`` of the original
    quantity once price reaches ``profit_pct`` above entry. The remainder
    (1 - sum(size_pct)) is the *runner*: it has no fixed target and rides a
    trailing stop, so a strong move is not capped. The stop only ever moves up:
    after tier ``move_stop_to_breakeven_after_tier`` fills it moves to entry
    (the trade becomes risk-free); after ``lock_stop_to_prior_tier_after_tier``
    it moves to the previous tier's price (locking realized gains). These knobs
    are config (not magic numbers) so the exit policy can be tuned and
    backtested. ``enabled=False`` falls back to the legacy single-leg TP/SL.
    """

    enabled: bool = True
    # Swing-width bracket: ~4% initial stop with multi-% scale-out tiers, sized for
    # hours-to-days holds off 1h/4h structure rather than an intraday scalp.
    initial_stop_loss_pct: float = 0.04
    take_profit_tiers: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"profit_pct": 0.030, "size_pct": 0.40},  # TP1
            {"profit_pct": 0.060, "size_pct": 0.30},  # TP2
        ]
    )
    # After these (1-based) tier indices fill, the stop ratchets up. 0 disables.
    move_stop_to_breakeven_after_tier: int = 1
    lock_stop_to_prior_tier_after_tier: int = 2
    # Runner (the unsold remainder) rides a trailing stop instead of a target.
    trail_runner: bool = True
    trail_pct: float = 0.03
    # When set, the runner trail distance is trail_atr_mult * ATR(RiskConfig.atr_interval,
    # i.e. 1h) instead of trail_pct (absolute price distance below the high-water
    # mark). ~2.5x 1h ATR gives the swing runner room to ride a multi-day move.
    trail_atr_mult: float | None = 2.5

    @property
    def runner_size_pct(self) -> float:
        """Fraction of the original quantity left on as the trailing runner."""
        return round(max(0.0, 1.0 - sum(t["size_pct"] for t in self.take_profit_tiers)), 10)


@dataclass(slots=True)
class CostConfig:
    """Material-change cycle tiers: spend the expensive deep-agent call only when
    the cycle state warrants it. The fast bracket monitor manages TP/SL between
    cycles, so skipping the LLM never leaves a position unmanaged.

    - FULL: full model + all subagents + MCP (entry signal, position near a
      bracket / outside the PnL band, first cycle of the UTC day, or a material
      price move).
    - REVIEW: cheap ``quiet_model`` + ``review_subagents`` only + no MCP (a quiet
      open position past the review interval).
    - SKIP: no LLM at all (flat book, no signal, no move) -> deterministic WAIT.
    """

    enabled: bool = True
    material_move_bps: float = 50.0
    position_review_band_pct: float = 0.8
    bracket_proximity_pct: float = 0.4
    review_interval_minutes: float = 180.0
    full_on_first_cycle_of_day: bool = True
    # "provider:model" for the REVIEW tier: a cheap mini so holding cycles run a
    # brief positions-only review at a fraction of FULL cost instead of either paying
    # full price to babysit or going dark. Operator-tunable; set None to make REVIEW
    # behave like SKIP (deterministic hold, no LLM).
    quiet_model: str | None = "openai:gpt-5.1-mini"
    # technical_analyst confirms structure/zone validity on a holding cycle; strategy
    # + risk_review weigh HOLD/ADJUST/CLOSE; reporting writes the audit line.
    review_subagents: list[str] = field(
        default_factory=lambda: ["technical_analyst", "strategy", "risk_review", "reporting"]
    )


@dataclass(slots=True)
class LiveConfig:
    enabled: bool = False
    venue_confirmed: bool = False
    exchange_venue: str = "adapter"
    capital_budget_usd: float = 100.0
    min_capital_budget_usd: float = 25.0
    max_capital_budget_usd: float = 100.0
    auto_orders_within_caps: bool = True
    # Promotion gate: a minimum sample of closed round trips before paper/testnet
    # results are trusted enough to consider a live-capital test (a single lucky
    # trade is not evidence of edge).
    promotion_min_closed_trades: int = 5


@dataclass(slots=True)
class ModelConfig:
    provider: str = "openai"
    model: str = "gpt-5.1"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    # Optional per-subagent overrides, e.g. {"news_research": "openai:gpt-5.1-mini"}.
    # Subagents without an entry inherit the supervisor's model.
    subagent_models: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    home: str = str(DEFAULT_HOME)
    database_path: str = str(DEFAULT_HOME / "agent.sqlite3")
    mode: str = "testnet"
    decision_interval_minutes: int = 60
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    cost: CostConfig = field(default_factory=CostConfig)


def parse_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_settings(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(parse_env_file(env_file))
    merged.update(dict(os.environ if environ is None else environ))
    return Settings(**merged)


def resolve_env_reference(value: str, *, environ: Mapping[str, str] | None = None) -> str | None:
    if not value.startswith("env:"):
        return value
    env = os.environ if environ is None else environ
    resolved = env.get(value.removeprefix("env:"))
    return resolved if resolved else None


def _merge_dataclass(default: Any, raw: dict[str, Any]) -> Any:
    data = asdict(default)
    for key, value in raw.items():
        if isinstance(default, RiskConfig) and key == "max_open_orders":
            key = "max_open_positions"
        if key not in data:
            continue
        if isinstance(data.get(key), dict) and isinstance(value, dict):
            nested_default = getattr(default, key)
            if hasattr(nested_default, "__dataclass_fields__"):
                data[key] = asdict(_merge_dataclass(nested_default, value))
            else:
                data[key] = {**data[key], **value}
        else:
            data[key] = value
    if isinstance(default, AppConfig):
        data["risk"] = RiskConfig(**data["risk"])
        data["backtest"] = BacktestConfig(**data["backtest"])
        data["live"] = LiveConfig(**data["live"])
        data["model"] = ModelConfig(**data["model"])
        data["strategy"] = StrategyConfig(**data["strategy"])
        data["sizing"] = SizingConfig(**data["sizing"])
        data["exits"] = ExitConfig(**data["exits"])
        data["cost"] = CostConfig(**data["cost"])
        return AppConfig(**data)
    if isinstance(default, RiskConfig):
        return RiskConfig(**data)
    if isinstance(default, BacktestConfig):
        return BacktestConfig(**data)
    if isinstance(default, LiveConfig):
        return LiveConfig(**data)
    if isinstance(default, ModelConfig):
        return ModelConfig(**data)
    if isinstance(default, StrategyConfig):
        return StrategyConfig(**data)
    if isinstance(default, SizingConfig):
        return SizingConfig(**data)
    if isinstance(default, ExitConfig):
        return ExitConfig(**data)
    if isinstance(default, CostConfig):
        return CostConfig(**data)
    raise TypeError(f"Unsupported config type: {type(default)!r}")


def config_path(home: Path = DEFAULT_HOME) -> Path:
    return home / "config.json"


def load_config(home: Path = DEFAULT_HOME) -> AppConfig:
    path = config_path(home)
    default = AppConfig(home=str(home), database_path=str(home / "agent.sqlite3"))
    if not path.exists():
        return default
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _merge_dataclass(default, raw)


def save_config(config: AppConfig) -> None:
    home = Path(config.home)
    home.mkdir(parents=True, exist_ok=True)
    config_path(home).write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def ensure_config(home: Path = DEFAULT_HOME) -> AppConfig:
    config = load_config(home)
    save_config(config)
    return config
