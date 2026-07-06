from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BINANCE_INSTALLED_SKILLS_ROOT = Path(".agents") / "skills"
BINANCE_READONLY_SKILL_SOURCE = Path("skills") / "binance-readonly"

READ_ONLY_CLI_COMMANDS: dict[str, set[str]] = {
    "crypto-market-rank": {
        "social-hype",
        "token-rank",
        "smart-money-inflow",
        "meme-rank",
        "address-pnl-rank",
    },
    "meme-rush": {"meme-rush", "topic-rush"},
    "query-address-info": {"positions"},
    "query-token-info": {"search", "meta", "dynamic", "kline"},
    "trading-signal": {"smart-money"},
}

# Binance Web3 skill CLIs are keyed by (chainId, contractAddress), not spot
# symbols. These are the canonical wrapped/pegged contracts for the majors we
# trade; pegged prices track spot 1:1. Chains are restricted to what the rank
# and signal commands support (BSC "56" and Solana "CT_501").
MAJOR_TOKEN_CONTRACTS: dict[str, dict[str, str]] = {
    "BTC": {
        "chainId": "56",
        "contractAddress": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",  # BTCB (Binance-Peg BTC)
        "wrappedSymbol": "BTCB",
    },
    "ETH": {
        "chainId": "56",
        "contractAddress": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",  # Binance-Peg ETH
        "wrappedSymbol": "ETH",
    },
    "BNB": {
        "chainId": "56",
        "contractAddress": "0xbb4CdB9CBd36B01bD1cBaEbF2De08d9173bc095c",  # WBNB
        "wrappedSymbol": "WBNB",
    },
    "SOL": {
        "chainId": "CT_501",
        "contractAddress": "So11111111111111111111111111111111111111112",  # Wrapped SOL
        "wrappedSymbol": "SOL",
    },
}


# Correct parameter shape per approved command. Returned in error responses so
# an agent that called a command wrongly can self-correct within the same cycle.
COMMAND_USAGE: dict[str, dict[str, str]] = {
    "crypto-market-rank": {
        "social-hype": '{"chainId":"56","targetLanguage":"en","timeRange":1}',
        "token-rank": '{"rankType":10,"chainId":"56","page":1,"size":20}',
        "smart-money-inflow": '{"chainId":"56","period":"24h"}',
        "meme-rank": '{"chainId":"56"}',
        "address-pnl-rank": '{"chainId":"CT_501","period":"30d","tag":"ALL","pageNo":1,"pageSize":25}',
    },
    "meme-rush": {
        "meme-rush": '{"chainId":"CT_501","rankType":10,"limit":20}',
        "topic-rush": '{"chainId":"CT_501","rankType":10,"sort":10}',
    },
    "query-address-info": {
        "positions": '{"address":"0x...","chainId":"56","offset":0}',
    },
    "query-token-info": {
        "search": '{"keyword":"<name|symbol|address>"}',
        "meta": '{"chainId":"56","contractAddress":"0x..."}',
        "dynamic": '{"chainId":"56","contractAddress":"0x..."}',
        "kline": '{"chainId":"56","contractAddress":"0x...","interval":"15min","limit":20}',
    },
    "trading-signal": {
        "smart-money": '{"chainId":"CT_501","page":1,"pageSize":20}',
    },
}

# Upstream kline intervals: 1s 1min 3min 5min 15min 30min 1h 2h 4h 6h 8h 12h
# 1d 3d 1w 1m (month!). LLMs habitually write minute intervals Binance-spot
# style ("15m"), which upstream rejects - or worse, "1m" which upstream reads
# as one MONTH. Normalize the minute shorthands explicitly.
_KLINE_INTERVAL_ALIASES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "1h",
    "1hour": "1h",
    "1day": "1d",
    "1week": "1w",
}

_BOGUS_PARAM_VALUES = {None, "", "undefined", "null", "None"}

_SPOT_QUOTE_SUFFIXES = ("USDT", "USDC", "FDUSD", "TUSD", "BUSD")


def resolve_major_token(value: object) -> dict[str, str] | None:
    """Map 'BTC', 'btc', or 'BTCUSDT' to its canonical (chainId, contract)."""
    if not isinstance(value, str):
        return None
    token = value.strip().upper()
    for suffix in _SPOT_QUOTE_SUFFIXES:
        if token.endswith(suffix) and token != suffix:
            token = token.removesuffix(suffix)
            break
    return MAJOR_TOKEN_CONTRACTS.get(token)


def _normalize_skill_params(
    skill_name: str,
    command: str,
    params: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Repair common LLM parameter mistakes instead of failing the call.

    Handles the failure modes observed in run logs: spot symbols passed to
    chain-keyed commands, literal "undefined"/"null" values, Binance-spot
    interval shorthands, and missing required defaults.
    """
    notes: list[str] = []
    normalized = {key: value for key, value in params.items() if value not in _BOGUS_PARAM_VALUES}
    if len(normalized) != len(params):
        notes.append("dropped empty/undefined parameter values")

    symbol_value = normalized.pop("symbol", None) or normalized.pop("token", None)
    normalized.pop("currency", None)  # not an upstream parameter on any command
    contract = resolve_major_token(symbol_value)

    if skill_name == "query-token-info":
        if command == "search":
            if symbol_value and "keyword" not in normalized:
                normalized["keyword"] = str(symbol_value)
                notes.append("mapped symbol to search keyword")
        elif command in {"meta", "dynamic", "kline"}:
            if contract and not (normalized.get("chainId") and normalized.get("contractAddress")):
                normalized["chainId"] = contract["chainId"]
                normalized["contractAddress"] = contract["contractAddress"]
                notes.append(
                    f"mapped {symbol_value} to chainId={contract['chainId']} "
                    f"contractAddress={contract['contractAddress']} ({contract['wrappedSymbol']})"
                )
            if command == "kline":
                interval = normalized.get("interval")
                if isinstance(interval, str) and interval in _KLINE_INTERVAL_ALIASES:
                    normalized["interval"] = _KLINE_INTERVAL_ALIASES[interval]
                    notes.append(f"normalized interval {interval} -> {normalized['interval']}")
    elif (skill_name, command) in {
        ("trading-signal", "smart-money"),
        ("crypto-market-rank", "smart-money-inflow"),
        ("crypto-market-rank", "social-hype"),
        ("crypto-market-rank", "token-rank"),
        ("crypto-market-rank", "meme-rank"),
    }:
        if contract and "chainId" not in normalized:
            normalized["chainId"] = contract["chainId"]
            notes.append(f"resolved chainId={contract['chainId']} from symbol {symbol_value}")
        if command == "social-hype":
            normalized.setdefault("targetLanguage", "en")
            normalized.setdefault("timeRange", 1)
        elif command == "smart-money-inflow":
            normalized.setdefault("period", "24h")
    elif skill_name == "meme-rush":
        topic = normalized.pop("topic", None)
        if topic and "keywords" not in normalized:
            normalized["keywords"] = [str(topic)]
            notes.append("mapped topic to keywords filter")
        normalized.setdefault("chainId", "CT_501")
        normalized.setdefault("rankType", 10)
        if command == "topic-rush":
            normalized.setdefault("sort", 10)
    elif (skill_name, command) == ("query-address-info", "positions"):
        if contract and "chainId" not in normalized:
            normalized["chainId"] = contract["chainId"]
        normalized.setdefault("offset", 0)

    return normalized, notes


REFERENCE_ONLY_SKILLS = {
    "binance-tokenized-securities-info",
    "query-token-audit",
}

EXECUTION_OR_AUTH_SENSITIVE_SKILLS = {
    "binance",
    "binance-agentic-wallet",
    "binance-sports-ai-analyzer",
    "fiat",
    "onchain-pay-open-api",
    "p2p",
    "payment-assistant",
    "square-post",
}


@dataclass(frozen=True, slots=True)
class BinanceSkillInfo:
    name: str
    path: Path
    description: str
    category: str
    commands: tuple[str, ...]

    @property
    def skill_file(self) -> Path:
        return self.path / "SKILL.md"


@dataclass(frozen=True, slots=True)
class BinanceSkillCommandResult:
    skill_name: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    params_used: str = ""
    normalization_notes: tuple[str, ...] = ()


class BinanceSkillRegistry:
    def __init__(
        self,
        root: str | Path = BINANCE_INSTALLED_SKILLS_ROOT,
        readonly_source: str | Path = BINANCE_READONLY_SKILL_SOURCE,
    ) -> None:
        self.root = Path(root)
        self.readonly_source = Path(readonly_source)

    def list_skills(self) -> list[BinanceSkillInfo]:
        if not self.root.exists():
            return []
        skills: list[BinanceSkillInfo] = []
        for path in sorted(item for item in self.root.iterdir() if item.is_dir()):
            skill_file = path / "SKILL.md"
            if not skill_file.exists():
                continue
            name, description = _read_frontmatter_summary(skill_file)
            skill_name = name or path.name
            commands = tuple(sorted(READ_ONLY_CLI_COMMANDS.get(skill_name, set())))
            skills.append(
                BinanceSkillInfo(
                    name=skill_name,
                    path=path,
                    description=description,
                    category=_category_for(skill_name),
                    commands=commands,
                )
            )
        return skills

    def skill(self, name: str) -> BinanceSkillInfo:
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        raise KeyError(f"unknown Binance skill: {name}")

    def read_only_skill_source_paths(self) -> list[str]:
        if not self.readonly_source.exists():
            return []
        return [_forward_slash_path(self.readonly_source)]

    def command_catalog(self) -> dict[str, list[str]]:
        return {
            skill.name: list(skill.commands)
            for skill in self.list_skills()
            if skill.commands
        }

    def run_read_only_cli(
        self,
        skill_name: str,
        command: str,
        params_json: str,
        *,
        timeout_seconds: int = 30,
    ) -> BinanceSkillCommandResult:
        skill = self.skill(skill_name)
        allowed = READ_ONLY_CLI_COMMANDS.get(skill.name)
        if not allowed:
            raise ValueError(f"skill {skill.name} does not expose an approved read-only CLI")
        # LLMs sometimes jam the skill name into the command argument
        # ("crypto-market-rank social-hype" or ".../social-hype"); strip a leading
        # skill-name prefix so the otherwise-valid call still runs.
        command = command.strip()
        for separator in (" ", "/"):
            prefix = f"{skill.name}{separator}"
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
                break
        if command not in allowed:
            raise ValueError(f"command {command} is not allowed for skill {skill.name}")
        parsed = json.loads(params_json)
        if not isinstance(parsed, dict):
            raise ValueError("params_json must decode to a JSON object")
        parsed, notes = _normalize_skill_params(skill.name, command, parsed)

        runtime_skill_path = self.readonly_source / skill.name
        if runtime_skill_path.exists():
            skill_path = runtime_skill_path
        else:
            skill_path = skill.path
        # Resolve to absolute paths: node interprets a relative script path
        # against the subprocess cwd, which would double the skill directory.
        skill_path = skill_path.resolve()
        script = skill_path / "scripts" / "cli.mjs"
        if not script.exists():
            raise FileNotFoundError(script)
        normalized_json = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        # Node writes UTF-8; without an explicit encoding, Windows decodes the
        # pipes as the locale codepage (cp1252) and crashes on multibyte JSON.
        completed = subprocess.run(
            ["node", str(script), command, normalized_json],
            cwd=skill_path,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return BinanceSkillCommandResult(
            skill_name=skill.name,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            params_used=normalized_json,
            normalization_notes=tuple(notes),
        )


# Web3 skill payloads carry 20-significant-figure numeric STRINGS
# (e.g. "price":"71.13274545547210661283") and chain-wide leaderboards with
# hundreds of rows. Both are pure token bloat for an LLM that only needs ~8
# significant figures and the top rows, so the raw stdout is compacted before it
# is handed back to the agent (the research stays identical; the precision and
# row-count noise does not).
_SKILL_MAX_ARRAY_ROWS = 30
_DECIMAL_STR = re.compile(r"^-?\d+\.\d+$")


def _compact_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.8g}")
    if isinstance(value, str) and _DECIMAL_STR.match(value.strip()):
        return f"{float(value):.8g}"
    if isinstance(value, list):
        rows = [_compact_value(item) for item in value[:_SKILL_MAX_ARRAY_ROWS]]
        if len(value) > _SKILL_MAX_ARRAY_ROWS:
            rows.append(f"...(+{len(value) - _SKILL_MAX_ARRAY_ROWS} more rows trimmed)")
        return rows
    if isinstance(value, dict):
        return {key: _compact_value(item) for key, item in value.items()}
    return value


def _compact_skill_stdout(text: str, max_chars: int = 8000) -> str:
    """Round long decimal strings to ~8 sig figs and cap long arrays; fall back
    to the raw slice when the payload is not the JSON we expect."""
    raw = text or ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw[:max_chars]
    try:
        return json.dumps(_compact_value(parsed), separators=(",", ":"))[:max_chars]
    except (TypeError, ValueError):
        return raw[:max_chars]


def run_binance_research_cli(skill_name: str, command: str, params_json: str) -> str:
    """Run an approved read-only Binance Skills Hub CLI command.

    Only allowlisted public-data Web3 research commands are permitted - never
    authenticated, wallet, payment, order, transfer, or cancellation skills.

    IMPORTANT: these CLIs are keyed by (chainId, contractAddress), NOT spot
    symbols. Canonical contracts for the trading majors (pegged price tracks
    spot 1:1):
    - BTC -> chainId "56",     contractAddress "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c" (BTCB)
    - ETH -> chainId "56",     contractAddress "0x2170Ed0880ac9A755fd29B2688956BD959F933F8"
    - BNB -> chainId "56",     contractAddress "0xbb4CdB9CBd36B01bD1cBaEbF2De08d9173bc095c" (WBNB)
    - SOL -> chainId "CT_501", contractAddress "So11111111111111111111111111111111111111112"
    Passing {"symbol":"BTCUSDT"} (or "btc") is also accepted: the tool maps
    majors to the contracts above automatically.

    Commands and exact parameter shapes:
    - query-token-info search  '{"keyword":"<name|symbol|address>"}'
    - query-token-info meta    '{"chainId":"56","contractAddress":"0x..."}'
    - query-token-info dynamic '{"chainId":"56","contractAddress":"0x..."}'
    - query-token-info kline   '{"chainId":"56","contractAddress":"0x...","interval":"15min","limit":20}'
      interval enum: 1s 1min 3min 5min 15min 30min 1h 2h 4h 6h 8h 12h 1d 3d 1w
      ("15m"-style shorthands are auto-corrected to "15min")
    - crypto-market-rank social-hype        '{"chainId":"56","targetLanguage":"en","timeRange":1}'
    - crypto-market-rank token-rank         '{"rankType":10,"chainId":"56","page":1,"size":20}'
    - crypto-market-rank smart-money-inflow '{"chainId":"56","period":"24h"}'  (chains 56/CT_501/8453)
    - crypto-market-rank meme-rank          '{"chainId":"56"}'
    - crypto-market-rank address-pnl-rank   '{"chainId":"CT_501","period":"30d","tag":"ALL"}'
    - meme-rush meme-rush   '{"chainId":"CT_501","rankType":10,"limit":20}'
    - meme-rush topic-rush  '{"chainId":"CT_501","rankType":10,"sort":10}'
    - trading-signal smart-money '{"chainId":"CT_501","page":1,"pageSize":20}'  (chains 56/CT_501)
    - query-address-info positions '{"address":"0x...","chainId":"56","offset":0}'

    Rank and signal commands return chain-wide leaderboards (often meme-heavy).
    Locate your token's row by contractAddress; a major being absent from a
    leaderboard is a neutral finding, not an error.
    """

    registry = BinanceSkillRegistry()
    try:
        result = registry.run_read_only_cli(skill_name, command, params_json)
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "skill_name": skill_name,
                "command": command,
                "allowed_commands": registry.command_catalog(),
                "usage": COMMAND_USAGE.get(skill_name, {}).get(command),
                "major_token_contracts": MAJOR_TOKEN_CONTRACTS,
            },
            sort_keys=True,
        )
    response: dict[str, Any] = {
        "ok": result.returncode == 0,
        "skill_name": result.skill_name,
        "command": result.command,
        "returncode": result.returncode,
        "params_used": result.params_used,
        "stdout": _compact_skill_stdout(result.stdout),
        "stderr": result.stderr[:4000],
    }
    if result.normalization_notes:
        response["normalization_notes"] = list(result.normalization_notes)
    if result.returncode != 0:
        response["usage"] = COMMAND_USAGE.get(skill_name, {}).get(command)
        response["major_token_contracts"] = MAJOR_TOKEN_CONTRACTS
    return json.dumps(response, sort_keys=True)


def _read_frontmatter_summary(path: Path) -> tuple[str | None, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None, ""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, ""
    frontmatter = parts[1].splitlines()
    name: str | None = None
    description_lines: list[str] = []
    in_description = False
    for line in frontmatter:
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped.removeprefix("name:").strip().strip("'\"")
            in_description = False
        elif stripped.startswith("description:"):
            raw = stripped.removeprefix("description:").strip()
            in_description = raw in {"|", ">"}
            if raw and not in_description:
                description_lines.append(raw.strip("'\""))
        elif in_description:
            if not line.startswith(" ") and stripped:
                in_description = False
            elif stripped:
                description_lines.append(stripped)
    return name, " ".join(description_lines).strip()


def _category_for(name: str) -> str:
    if name in READ_ONLY_CLI_COMMANDS:
        return "read_only_cli"
    if name in REFERENCE_ONLY_SKILLS:
        return "reference_only"
    if name in EXECUTION_OR_AUTH_SENSITIVE_SKILLS:
        return "execution_or_auth_sensitive"
    return "unclassified"


def _forward_slash_path(path: Path) -> str:
    return path.as_posix()
