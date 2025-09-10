from __future__ import annotations

from pathlib import Path

from util.utils import _find_project_root


def test_find_project_root_fallback(tmp_path: Path) -> None:
    # tmp_path/a/b のような構造（上流に .git/pyproject.toml/configs が無い）では
    # フォールバックで start.parent.parent を返す
    a = tmp_path / "a" / "b"
    a.mkdir(parents=True)
    start = a
    got = _find_project_root(start)
    assert got == start.parent.parent
