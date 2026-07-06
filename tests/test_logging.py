from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.logging import configure_logging, get_logger, shutdown_logging


class LoggingTest(unittest.TestCase):
    def test_configure_logging_writes_rotating_file(self) -> None:
        with TemporaryDirectory() as tmp:
            logger = configure_logging(tmp, level="INFO", log_to_stderr=False, log_to_file=True)
            get_logger("test").info("hello live logs")
            for handler in logger.handlers:
                handler.flush()

            log_path = Path(tmp) / "logs" / "trading_agent.log"
            text = log_path.read_text(encoding="utf-8")
            shutdown_logging()

        self.assertIn("hello live logs", text)
        self.assertIn("trading_agent.test", text)


if __name__ == "__main__":
    unittest.main()
