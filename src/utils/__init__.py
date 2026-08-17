"""Logging utilities."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_error(name: str, reason: str) -> None:
    """Surface a non-fatal error from a model loader."""
    get_logger("models").warning("model_error name=%s reason=%s", name, reason)


# Backward-compatibility alias (used by older code paths).
log_fallback = log_error