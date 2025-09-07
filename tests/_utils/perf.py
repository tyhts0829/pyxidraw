from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

BASELINE_DIR = Path("tests/_snapshots/perf")


def _bool_env(name: str) -> bool:
    v = os.getenv(name, "0")
    return v in ("1", "true", "TRUE", "True")


def load_baseline(name: str) -> float | None:
    """Load a baseline value by `name`.

    Returns seconds (float) or None if not found.
    """
    path = BASELINE_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        v = float(data.get("seconds"))
        return v
    except Exception:
        return None


def maybe_update_baseline(name: str, seconds: float) -> None:
    """Update baseline if PXD_UPDATE_SNAPSHOTS=1 is set.

    Writes a small JSON file that stores the timing in seconds along with metadata.
    """
    if not _bool_env("PXD_UPDATE_SNAPSHOTS"):
        return
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{name}.json"
    payload = {
        "seconds": seconds,
        "note": "Auto-updated by tests when PXD_UPDATE_SNAPSHOTS=1",
        "env": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
