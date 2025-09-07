from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Iterable

SNAPSHOT_DIR = Path("tests/_snapshots")


def _bool_env(name: str) -> bool:
    v = os.getenv(name, "0")
    return v in ("1", "true", "TRUE", "True")


def snapshots_enabled() -> bool:
    """Return True if snapshot updating is enabled via env."""
    return _bool_env("PXD_UPDATE_SNAPSHOTS")


def path_for(module: str, test_name: str) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR / f"{module}__{test_name}.snap"


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def assert_or_update(path: Path, new_lines: list[str]) -> None:
    """Assert that `new_lines` match snapshot at `path` or update if env set.

    - If PXD_UPDATE_SNAPSHOTS=1, (re)write snapshot and return.
    - Otherwise, compare; on mismatch, raise AssertionError with a unified diff.
    """
    if snapshots_enabled() or not path.exists():
        write_lines(path, new_lines)
        return

    old_lines = read_lines(path)
    if old_lines == new_lines:
        return

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=str(path),
        tofile="<generated>",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    raise AssertionError(
        "Snapshot mismatch. To accept changes, set PXD_UPDATE_SNAPSHOTS=1 and re-run tests.\n"
        + diff_text
    )
