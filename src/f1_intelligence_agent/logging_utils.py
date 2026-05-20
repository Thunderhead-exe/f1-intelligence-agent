"""Logging helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise stdout logger for app and CLI runs."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=False,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)

