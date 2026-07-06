from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentEvent:
    """One observable moment of agent activity, rendered by the REPL."""

    kind: str  # lifecycle | supervisor | token | update | snapshot | evidence |
    #            proposal | risk_decision | order | reconcile | skill_call | error | info
    agent: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe fan-in queue between the agent thread and the UI thread."""

    def __init__(self, maxsize: int = 10_000) -> None:
        self._queue: queue.Queue[AgentEvent] = queue.Queue(maxsize=maxsize)

    def emit(self, event: AgentEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest to keep the stream live rather than blocking trading.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(event)

    def emit_now(self, kind: str, agent: str, data: dict[str, Any]) -> None:
        self.emit(AgentEvent(kind=kind, agent=agent, data=data))

    def drain(self, limit: int = 200) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        while len(events) < limit:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events
