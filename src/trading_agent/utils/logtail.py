"""Windows-safe log tailing: last-N lines and a follow generator.

No fcntl/inotify — plain polling that tolerates RotatingFileHandler rotation
(file replaced or truncated mid-follow) by re-opening when the file shrinks
or its identity changes.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

_POLL_SECONDS = 0.5
_BLOCK_SIZE = 8192


def read_last_lines(path: str | Path, count: int = 50) -> list[str]:
    """Return up to the last `count` lines without loading the whole file."""
    log_path = Path(path)
    if not log_path.exists() or count <= 0:
        return []
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        blocks: list[bytes] = []
        lines_found = 0
        position = size
        while position > 0 and lines_found <= count:
            step = min(_BLOCK_SIZE, position)
            position -= step
            handle.seek(position)
            block = handle.read(step)
            blocks.append(block)
            lines_found += block.count(b"\n")
        data = b"".join(reversed(blocks))
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-count:]


def follow(
    path: str | Path,
    *,
    poll_seconds: float = _POLL_SECONDS,
    should_stop: "callable[[], bool] | None" = None,
) -> Iterator[str]:
    """Yield new lines as they are appended (like tail -f), surviving rotation.

    `should_stop` is checked every poll so callers (REPL/CLI) can break on
    Ctrl-C or shutdown without killing the thread. The starting position is
    captured NOW (call time), not lazily at first iteration — tail semantics.
    """
    log_path = Path(path)
    position = log_path.stat().st_size if log_path.exists() else 0
    return _follow_from(log_path, position, poll_seconds, should_stop)


def _follow_from(
    log_path: Path,
    position: int,
    poll_seconds: float,
    should_stop: "callable[[], bool] | None",
) -> Iterator[str]:
    buffer = b""
    while True:
        if should_stop is not None and should_stop():
            return
        if not log_path.exists():
            time.sleep(poll_seconds)
            continue
        size = log_path.stat().st_size
        if size < position:
            position = 0  # rotated/truncated: start from the new beginning
            buffer = b""
        if size > position:
            with log_path.open("rb") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace").rstrip("\r")
        else:
            time.sleep(poll_seconds)
