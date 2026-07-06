from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum

from trading_agent.repl.events import EventBus
from trading_agent.utils.binance_skills import READ_ONLY_CLI_COMMANDS


class AgentState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


class AgentLifecycleManager:
    """NemoClaw-style runtime controls for the always-on agent loop.

    - Lifecycle state machine with explicit transitions
    - Pause/resume gate checked between cycles (never mid-order)
    - Stop flag for graceful shutdown
    - Skill execution policy: only allowlisted read-only commands pass
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._pause_gate = threading.Event()
        self._stop_flag = threading.Event()
        self._pause_gate.set()
        self._state = AgentState.IDLE

    @property
    def state(self) -> AgentState:
        return self._state

    def _transition(self, state: AgentState) -> None:
        with self._lock:
            previous, self._state = self._state, state
        self._bus.emit_now("lifecycle", "lifecycle", {"from": previous, "to": state})

    def starting(self) -> None:
        self._stop_flag.clear()
        self._pause_gate.set()
        self._transition(AgentState.STARTING)

    def running(self) -> None:
        self._transition(AgentState.RUNNING)

    def pause(self) -> None:
        if self._state is AgentState.RUNNING:
            self._pause_gate.clear()
            self._transition(AgentState.PAUSED)

    def resume(self) -> None:
        if self._state is AgentState.PAUSED:
            self._pause_gate.set()
            self._transition(AgentState.RUNNING)

    def stop(self) -> None:
        if self._state not in {AgentState.STOPPED, AgentState.IDLE}:
            self._transition(AgentState.STOPPING)
        self._stop_flag.set()
        self._pause_gate.set()  # unblock a paused loop so it can exit

    def stopped(self) -> None:
        self._transition(AgentState.STOPPED)

    def stop_requested(self) -> bool:
        return self._stop_flag.is_set()

    def wait_or_die(self, poll_seconds: float = 0.5) -> bool:
        """Block while paused; return False once stop is requested.

        Call between cycles (and during interval sleeps) so pause takes effect
        at safe boundaries only.
        """
        while not self._pause_gate.wait(timeout=poll_seconds):
            if self._stop_flag.is_set():
                return False
        return not self._stop_flag.is_set()

    def sleep_or_die(
        self,
        seconds: float,
        poll_seconds: float = 0.5,
        *,
        on_tick: Callable[[], None] | None = None,
        tick_seconds: float = 0.0,
    ) -> bool:
        """Interruptible sleep honoring pause/stop. False means stop.

        If ``on_tick`` is given it is invoked roughly every ``tick_seconds`` while
        sleeping — used to run the fast bracket monitor (TP/SL) between cycles so
        exits act within seconds rather than only at the next decision cycle.
        """
        remaining = seconds
        since_tick = 0.0
        while remaining > 0:
            if not self.wait_or_die(poll_seconds):
                return False
            step = min(poll_seconds, remaining)
            if self._stop_flag.wait(timeout=step):
                return False
            remaining -= step
            since_tick += step
            if on_tick is not None and tick_seconds > 0 and since_tick >= tick_seconds:
                since_tick = 0.0
                on_tick()
        return not self._stop_flag.is_set()

    @staticmethod
    def check_skill_allowed(skill_name: str, command: str) -> bool:
        """Policy gate: only allowlisted read-only skill commands may run."""
        return command in READ_ONLY_CLI_COMMANDS.get(skill_name, set())
