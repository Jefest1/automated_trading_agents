from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.utils.logtail import follow, read_last_lines


class ReadLastLinesTest(unittest.TestCase):
    def test_returns_last_n_lines(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.log"
            path.write_text("\n".join(f"line-{i}" for i in range(200)) + "\n", encoding="utf-8")
            lines = read_last_lines(path, 3)
        self.assertEqual(lines, ["line-197", "line-198", "line-199"])

    def test_short_file_returns_everything(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.log"
            path.write_text("a\nb\n", encoding="utf-8")
            self.assertEqual(read_last_lines(path, 50), ["a", "b"])

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(read_last_lines("does-not-exist.log", 10), [])


class FollowTest(unittest.TestCase):
    def test_follow_yields_appended_lines_and_survives_truncation(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.log"
            path.write_text("old\n", encoding="utf-8")
            collected: list[str] = []
            stops = {"flag": False}

            generator = follow(path, poll_seconds=0.01, should_stop=lambda: stops["flag"])

            with path.open("a", encoding="utf-8") as handle:
                handle.write("new-1\nnew-2\n")
            collected.append(next(generator))
            collected.append(next(generator))

            # rotation: file truncated and rewritten from scratch
            path.write_text("after-rotate\n", encoding="utf-8")
            collected.append(next(generator))

            stops["flag"] = True
            self.assertEqual(list(generator), [])

        self.assertEqual(collected, ["new-1", "new-2", "after-rotate"])


if __name__ == "__main__":
    unittest.main()
