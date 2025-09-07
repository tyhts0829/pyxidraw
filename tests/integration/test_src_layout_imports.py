from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
# What this tests (TEST_HARDENING_PLAN.md §構成変更)
# - Simulate a src/ layout by symlinking top-level packages under tmp/src and ensure
#   imports resolve without relying on repo-root or CWD quirks. Runs in a subprocess
#   to avoid polluting the parent interpreter's sys.modules.
def test_src_layout_style_imports_with_symlink(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    real_src = repo_root / "src"
    if real_src.exists():
        src_dir = real_src
    else:
        src_dir = tmp_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        top_level = ["api", "engine", "effects", "shapes", "util", "common"]
        for name in top_level:
            target = repo_root / name
            if not target.exists():
                continue
            link = src_dir / name
            try:
                os.symlink(target, link, target_is_directory=True)
            except (FileExistsError, OSError):
                import shutil

                shutil.copytree(target, link, dirs_exist_ok=True)

    script = (
        "import sys, importlib\n"
        "from pathlib import Path\n"
        f"src = Path(r'{str(src_dir)}')\n"
        f"repo = Path(r'{str(repo_root)}')\n"
        "sys.path[:] = [str(src)] + [p for p in sys.path if Path(p).resolve() != repo.resolve()]\n"
        "m = importlib.import_module('api')\n"
        "assert hasattr(m, 'G') and hasattr(m, 'E') and hasattr(m, 'run')\n"
    )

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
