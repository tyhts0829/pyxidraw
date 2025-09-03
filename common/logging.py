"""
Lightweight logging utilities for the project (Proposal 7).

Default behavior: modules import logging and obtain a logger via
`logging.getLogger(__name__)`. This helper ensures a sane default
configuration if the application hasn't configured logging yet.
"""

from __future__ import annotations

import logging
from typing import Literal


def setup_default_logging(level: int | str = "INFO") -> None:
    """Setup a minimal logging configuration once.

    - No-op if root logger already has handlers
    - Intended to be called from top-level runners/CLIs
    """
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)

    root = logging.getLogger()
    if root.handlers:
        # Assume the app has configured logging
        return
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


__all__ = ["setup_default_logging"]

