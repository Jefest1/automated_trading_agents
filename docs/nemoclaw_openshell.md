# NemoClaw/OpenShell Runtime Notes

The MVP keeps shell execution outside Python dependency management. NemoClaw/OpenShell is the WSL/Linux runtime path for governed 24/7 shell access, but no local shell MCP server is enabled in the current implementation.

## Target

- Runtime host: WSL/Linux.
- Project path: the trading agent workspace only.
- Logs: `.trading_agent/logs`.
- Python dependencies: managed with `uv`; do not install NemoClaw/OpenShell with `uv add`.
- MCP mode: hosted public HTTP MCP clients only for now.
- Future custom trading-data tools: standalone FastAPI service with a standard service layout, not an in-process FastMCP module.
- Env validation: `src/trading_agent/core/config.py`; do not use `load_dotenv`.

## Research Notes

NVIDIA describes NemoClaw as an open source reference stack for running always-on AI agents inside OpenShell sandboxes. The important pieces for this trading agent are lifecycle management, sandboxing, model routing, network policy, filesystem/process controls, observability, and skill execution.

OpenShell policy should be deny-by-default:

- Network egress starts blocked and allows only named endpoints needed for research and package maintenance.
- Filesystem access is scoped to this workspace plus `.trading_agent/logs`.
- Process execution is restricted to diagnostics, tests, and controlled CLI invocations.
- Inference and tool use are policy-visible.
- Secrets are kept out of the agent shell. NemoClaw credential storage is designed so sandboxed agents see placeholders while the gateway substitutes credentials at egress.

Initial install target is WSL/Linux with the official installer:

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

Do not run this from Python dependency management and do not enable shell tools until a workspace-specific policy is reviewed.

## Shell Policy Profile

Allow:

- `uv run python -m unittest discover -s tests`
- `uv run python -m trading_agent.cli agent status`
- `uv run python -m trading_agent.cli agent introduce`
- `uv run python -m trading_agent.cli agent once --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT`
- read-only diagnostics such as listing files, checking process status, and inspecting logs inside this workspace

Block:

- secret reads from `.env`, `.env.test`, shell history, SSH keys, cloud credentials, browser profiles, wallet files, and exchange API key locations
- destructive filesystem commands outside the workspace
- network credential exfiltration
- Binance or exchange order placement/cancellation commands
- commands that start live trading or write private keys

## Runtime Flags

The Python service validates these flags through `Settings`:

- `NEMOCLAW_ENABLED`: records whether the operator intends to run inside NemoClaw/OpenShell.
- `NEMOCLAW_SHELL_ENABLED`: must stay `false` until the shell policy is reviewed and active.
- `NEMOCLAW_SANDBOX`: sandbox name, default `trading-agent`.
- `NEMOCLAW_POLICY_PROFILE`: default `trading-agent-readonly-research`.
- `NEMOCLAW_POLICY_EXPLAIN_PATH`: path where policy explanations can be exported for agent-readable context.

Future implementation path:

1. Onboard the workspace into NemoClaw/OpenShell on WSL/Linux.
2. Export an agent-readable policy explanation into `.trading_agent/nemoclaw/POLICY.md`.
3. Add a shell tool adapter that refuses to run unless `NEMOCLAW_ENABLED=true`, `NEMOCLAW_SHELL_ENABLED=true`, and the policy explanation exists.
4. Keep all trade execution unavailable to shell tools. The only allowed trading path remains `TradeIntent` -> `RiskGovernor` -> `PaperExecutionEngine`.

## MCP Integration

Copy `config/mcp_servers.example.json` to `.trading_agent/mcp_servers.json` and enable only hosted HTTP MCP servers needed for the current run.

The current MCP loader ignores `stdio` entries. LLM agents receive hosted research/documentation tools only. Execution-capable Binance skills or MCP servers are not part of the MVP.

The initial hosted examples are:

- Context7 docs MCP: `https://mcp.context7.com/mcp`
- GitMCP LangGraph context: `https://gitmcp.io/langchain-ai/langgraph`
- GitMCP Deep Agents context: `https://gitmcp.io/langchain-ai/deepagents`
- GitMCP Binance Skills Hub context: `https://gitmcp.io/binance/binance-skills-hub`

## Sources

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- LangChain MCP adapters: https://docs.langchain.com/oss/python/langchain/mcp
- NVIDIA NemoClaw/OpenShell: https://www.nvidia.com/en-us/ai/nemoclaw/
- NVIDIA NemoClaw GitHub: https://github.com/NVIDIA/NemoClaw
- NVIDIA NemoClaw docs: https://docs.nvidia.com/nemoclaw/latest/
- NemoClaw security best practices: https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/security/best-practices
- NemoClaw credential storage: https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/security/credential-storage
- NemoClaw network policy customization: https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/network-policy/customize-network-policy
