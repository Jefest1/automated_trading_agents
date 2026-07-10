"""Streaming/observability helpers for deep-agent invocations."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from trading_agent.core.logging import get_logger
from trading_agent.graph.state import EventCallback

LOGGER = get_logger("streaming")


class ToolCallLogger(BaseCallbackHandler):
    """Logs every tool invocation by name across the supervisor and all
    subagents. Callbacks propagate into nested subgraphs and fire in both the
    streaming and non-streaming paths, so this is the reliable way to see which
    tools (built-in vs attached MCP servers) the agents actually call; the
    deepagents subagent namespaces do not carry the subagent name into stream
    chunks, so stream-based attribution alone is unreliable."""

    def __init__(self, mcp_tool_names: set[str] | None = None) -> None:
        self.mcp_tool_names = mcp_tool_names or set()

    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "?"
        kind = "mcp_tool" if name in self.mcp_tool_names else "tool"
        preview = " ".join(str(input_str or "").split())[:160]
        LOGGER.info("%s call name=%s input=%s", kind, name, preview)

_KNOWN_AGENT_NAMES = {
    "supervisor",
    "consult_agents",
    "trading-supervisor",
    "market_research",
    "news_research",
    "onchain_research",
    "strategy",
    "risk_review",
    "reporting",
}
_STRUCTURAL_NAMESPACE_NAMES = {"tools", "model"}


def stream_deep_agent(
    agent: Any,
    payload: dict[str, Any],
    invoke_config: dict[str, Any],
    emit: EventCallback,
) -> Any:
    """Stream the deep agent run, forwarding chunks as observability events.

    Returns the final state (same shape as ``invoke``) so callers do not
    need to care whether streaming was active. The final state is captured
    from "values" chunks: the root graph emits its full state after every
    superstep, and the last one is exactly what ``invoke`` would return.
    Deriving it from "updates" deltas or ``get_state`` is unreliable here
    (nested invocation shares the checkpointer and thread) and surfaces as
    empty supervisor responses.
    """
    final_state: Any = None
    supervisor_message = ""
    for chunk in agent.stream(
        payload,
        config=invoke_config,
        stream_mode=["updates", "messages", "values"],
        subgraphs=True,
    ):
        final_state, supervisor_message = _handle_chunk(chunk, emit, final_state, supervisor_message)
    if final_state is not None:
        return final_state
    if supervisor_message:
        # Expected path: deepagents + subgraphs does not surface a root-namespace
        # "values" chunk, so the streamed supervisor message IS our reliable result
        # (the decision JSON is parsed from it). This is success, not a fault.
        LOGGER.debug("deep agent stream: using streamed supervisor output (no root values chunk)")
        return _message_state(supervisor_message)
    LOGGER.warning("deep agent stream produced no supervisor output; falling back to checkpoint state")
    if hasattr(agent, "get_state"):
        try:
            return agent.get_state(invoke_config).values
        except Exception as exc:
            LOGGER.warning("deep agent checkpoint fallback failed: %s", exc)
    return {}


async def astream_deep_agent(
    agent: Any,
    payload: dict[str, Any],
    invoke_config: dict[str, Any],
    emit: EventCallback,
) -> Any:
    """Async variant of stream_deep_agent."""
    final_state: Any = None
    supervisor_message = ""
    async for chunk in agent.astream(
        payload,
        config=invoke_config,
        stream_mode=["updates", "messages", "values"],
        subgraphs=True,
    ):
        final_state, supervisor_message = _handle_chunk(chunk, emit, final_state, supervisor_message)
    if final_state is not None:
        return final_state
    if supervisor_message:
        # Expected path: deepagents + subgraphs does not surface a root-namespace
        # "values" chunk, so the streamed supervisor message IS our reliable result
        # (the decision JSON is parsed from it). This is success, not a fault.
        LOGGER.debug("deep agent stream: using streamed supervisor output (no root values chunk)")
        return _message_state(supervisor_message)
    LOGGER.warning("deep agent stream produced no supervisor output; falling back to checkpoint state")
    if hasattr(agent, "aget_state"):
        try:
            return (await agent.aget_state(invoke_config)).values
        except Exception as exc:
            LOGGER.warning("deep agent checkpoint fallback failed: %s", exc)
    return {}


def _handle_chunk(
    chunk: Any,
    emit: EventCallback,
    final_state: Any,
    supervisor_message: str,
) -> tuple[Any, str]:
    namespace, mode, data = split_stream_chunk(chunk)
    agent_name = agent_name_from_namespace(namespace)
    if mode == "messages":
        text = message_chunk_text(data)
        if not text:
            return final_state, supervisor_message
        # The REPL shows model text from every agent/subagent. Tool results are
        # still log-only (watch with `/logs follow` or
        # `trading-agent logs --follow`).
        if _is_tool_message(data):
            LOGGER.info("tool result agent=%s: %s", agent_name, text[:1000])
            return final_state, supervisor_message
        if _is_supervisor_stream(namespace, agent_name):
            supervisor_message += text
        emit("token", agent_name, {"text": text})
    elif mode == "values":
        if not namespace and isinstance(data, dict):
            return data, supervisor_message
    elif mode == "updates" and isinstance(data, dict):
        for node_name, update in data.items():
            if str(node_name) == "model" and _is_supervisor_stream(namespace, agent_name):
                supervisor_message = supervisor_message_from_update(update) or supervisor_message
            if namespace and str(node_name) == "model" and not _update_has_tool_calls(update):
                text = supervisor_message_from_update(update)
                if text:
                    emit("token", agent_name, {"text": text})
            if update is None:
                continue  # middleware no-ops just clutter the log
            if str(node_name) == "tools":
                # Tool results are already logged once from the messages
                # stream; the updates echo (root + subgraph) tripled the log.
                continue
            LOGGER.info(
                "agent update agent=%s node=%s summary=%s",
                agent_name if namespace else str(node_name),
                node_name,
                summarize_update(update, limit=1000),
            )
    return final_state, supervisor_message


def _is_tool_message(data: Any) -> bool:
    """True when a "messages" chunk carries a ToolMessage (tool output), not
    model-generated text. Checked via the message type and the metadata's
    originating node."""
    message = data[0] if isinstance(data, tuple) and data else data
    metadata = data[1] if isinstance(data, tuple) and len(data) > 1 and isinstance(data[1], dict) else {}
    if str(metadata.get("langgraph_node", "")) == "tools":
        return True
    if isinstance(message, dict):
        return message.get("role") == "tool" or message.get("type") == "tool" or "tool_call_id" in message
    if getattr(message, "type", None) == "tool":
        return True
    return getattr(message, "tool_call_id", None) is not None


def _is_supervisor_stream(namespace: tuple[str, ...], agent_name: str) -> bool:
    if not namespace:
        return True
    return agent_name in {"supervisor", "consult_agents", "trading-supervisor"}


def _message_state(content: str) -> dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": content.strip()}]}


def split_stream_chunk(chunk: Any) -> tuple[tuple[str, ...], str, Any]:
    """Normalize langgraph stream chunks across shapes.

    With subgraphs=True and multiple stream modes, chunks arrive as
    (namespace, mode, data); with a single mode as (namespace, data); plain
    dicts appear when neither option is active.
    """
    if isinstance(chunk, tuple):
        if len(chunk) == 3:
            namespace, mode, data = chunk
            return tuple(namespace or ()), str(mode), data
        if len(chunk) == 2:
            first, second = chunk
            if isinstance(first, tuple):
                return tuple(first), "updates", second
            return (), str(first), second
    return (), "updates", chunk


def agent_name_from_namespace(namespace: tuple[str, ...]) -> str:
    if not namespace:
        return "supervisor"
    # Namespace entries look like "subagent_name:<task-id>". Prefer named
    # trading agents over structural nodes such as "tools".
    names = [str(part).split(":", 1)[0] for part in namespace]
    for name in reversed(names):
        if name in _KNOWN_AGENT_NAMES:
            return name
    for name in reversed(names):
        if name not in _STRUCTURAL_NAMESPACE_NAMES:
            return name
    return names[-1]


def message_chunk_text(data: Any) -> str:
    message = data[0] if isinstance(data, tuple) and data else data
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def summarize_update(update: Any, limit: int = 300) -> str:
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            tool_calls = getattr(last, "tool_calls", None)
            if tool_calls:
                names = ", ".join(str(call.get("name", "?")) for call in tool_calls)
                return f"tool calls: {names}"
            text = message_chunk_text(last)
            if text:
                return text[:limit]
    return str(update)[:limit]


def _update_has_tool_calls(update: Any) -> bool:
    if not isinstance(update, dict):
        return False
    messages = update.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return True
    if isinstance(last, dict):
        return bool(last.get("tool_calls"))
    return False


def supervisor_message_from_update(update: Any) -> str:
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            text = message_chunk_text(last)
            if text:
                return text
    return ""


def _content_to_text(content: Any) -> str:
    """Flatten a message's content into plain text.

    The Responses API (and multimodal chat) return content as a LIST of typed
    blocks, e.g. [{"type": "text", "text": "...```json ... ```"}]. str()-ing the
    list yields a Python repr with escaped newlines, which hides the fenced
    decision block from the parser. Concatenate the text parts instead so the
    decision contract survives the Chat-Completions -> Responses API switch.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text is None and block.get("type") in {"text", "output_text"}:
                    text = block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def extract_agent_message(response: Any) -> str:
    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content")
                if content:
                    return _content_to_text(content)
            content = getattr(last, "content", None)
            if content:
                return _content_to_text(content)
        for key in ("content", "output", "message"):
            value = response.get(key)
            if value:
                return _content_to_text(value)
    content = getattr(response, "content", None)
    if content:
        return _content_to_text(content)
    return str(response)
