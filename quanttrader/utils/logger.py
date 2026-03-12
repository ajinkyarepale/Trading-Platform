"""
utils/logger.py
===============
Centralized logging setup for the entire platform.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger with console + file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Log level override (default reads from settings).

    Returns:
        Configured Logger instance.
    """
    from config.settings import get_settings

    settings = get_settings()
    effective_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(effective_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(effective_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = settings.log_dir / "quanttrader.log"
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
