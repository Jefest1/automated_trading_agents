"""Persistent SQLite checkpointing for the deep agent's conversation memory.

The deep agent (supervisor + subagents) keeps its thread history in
<HOME>/checkpoints.sqlite3 so memory survives restarts and is shared between
supervised cycles and operator /chat sessions on the same thread id.

aiosqlite connections are bound to the event loop that created them, and the
REPL runs cycles and chat on different threads, each entering asyncio via
asyncio.run(). The saver therefore MUST be opened per top-level async call
(async with factory.open()) and never cached on the runtime — persistence
comes from the file, not the object.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB_FILENAME = "checkpoints.sqlite3"


class CheckpointerFactory:
    def __init__(self, home: str | Path) -> None:
        self.path = Path(home) / CHECKPOINT_DB_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def open(self) -> AsyncIterator[AsyncSqliteSaver]:
        async with AsyncSqliteSaver.from_conn_string(str(self.path)) as saver:
            await saver.setup()
            yield saver

    @contextmanager
    def open_sync(self) -> Iterator[SqliteSaver]:
        with SqliteSaver.from_conn_string(str(self.path)) as saver:
            saver.setup()
            yield saver
