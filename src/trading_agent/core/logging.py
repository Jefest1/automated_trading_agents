from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "trading_agent"
# Separate root (not a child of trading_agent): chat traffic goes ONLY to
# logs/chat.log, never interleaved into the main agent log.
CHAT_LOGGER_NAME = "trading_agent_chat"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
logging.getLogger(CHAT_LOGGER_NAME).addHandler(logging.NullHandler())


def configure_logging(
    home: str | Path,
    *,
    level: str = "INFO",
    log_to_stderr: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_level_value(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if log_to_stderr:
        # On Windows the console stream often defaults to cp1252, which mangles
        # non-ASCII log content (≈, en dashes, etc.). Pin UTF-8 where supported.
        reconfigure = getattr(sys.stderr, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(_level_value(level))
        logger.addHandler(stderr_handler)

    if log_to_file:
        log_dir = Path(home) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "trading_agent.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_level_value(level))
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    _configure_chat_logging(home, level=level, log_to_file=log_to_file)

    logger.debug("logging configured", extra={})
    return logger


def _configure_chat_logging(home: str | Path, *, level: str, log_to_file: bool) -> None:
    """Operator chat transcript log: <home>/logs/chat.log, file-only (the REPL
    already renders chat to the console)."""
    chat_logger = logging.getLogger(CHAT_LOGGER_NAME)
    chat_logger.setLevel(_level_value(level))
    chat_logger.propagate = False
    for handler in list(chat_logger.handlers):
        chat_logger.removeHandler(handler)
        handler.close()
    if log_to_file:
        log_dir = Path(home) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "chat.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handler.setLevel(_level_value(level))
        chat_logger.addHandler(handler)
    if not chat_logger.handlers:
        chat_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    suffix = f".{name}" if name else ""
    return logging.getLogger(f"{LOGGER_NAME}{suffix}")


def get_chat_logger() -> logging.Logger:
    return logging.getLogger(CHAT_LOGGER_NAME)


def shutdown_logging() -> None:
    for logger_name in (LOGGER_NAME, CHAT_LOGGER_NAME):
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.addHandler(logging.NullHandler())


def _level_value(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)
