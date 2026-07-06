"""Run coroutines on a persistent per-thread event loop.

`asyncio.run(coro)` creates a FRESH event loop and CLOSES it on every call. On
Windows' ProactorEventLoop that means any httpx/anyio async client created during
the call (LLM clients, MCP HTTP sessions) is finalized AFTER its loop is gone, so
its `AsyncClient.aclose()` raises `RuntimeError: Event loop is closed` and floods
stderr with "Task exception was never retrieved" tracebacks.

Reusing ONE event loop per thread for the lifetime of the session fixes this: the
clients live on a loop that stays open, their cleanup callbacks run on a live
loop, and there is no per-call close→GC race. There are only a handful of threads
in this app (the REPL/main thread, the agent-loop thread, LangGraph's sync-node
executor threads), so this is one open loop per thread, closed implicitly at
process exit. Loops are thread-confined, so a thread-local is the correct scope.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")

_local = threading.local()


def get_thread_loop() -> asyncio.AbstractEventLoop:
    """The calling thread's persistent event loop, created on first use."""
    loop: asyncio.AbstractEventLoop | None = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _local.loop = loop
    return loop


def run_coro_blocking(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion on this thread's persistent loop.

    Drop-in replacement for ``asyncio.run`` that does NOT close the loop. Must not
    be called while a loop is already running on this thread (the caller checks).
    """
    return get_thread_loop().run_until_complete(coro)
