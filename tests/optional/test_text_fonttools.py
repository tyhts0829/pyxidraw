from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fontTools")
pytest.importorskip("fontPens")
pytest.importorskip("numba")

from shapes.text import Text  # noqa: E402

# What this tests (TEST_HARDENING_PLAN.md §Optional)
# - With real fontTools/fontPens/numba and a system font available, shapes.Text generates
#   non-empty geometry; otherwise the test is skipped.


def _find_any_system_font() -> Path | None:
    # Allow overriding via environment (CI stabilization)
    import os

    env_font = os.getenv("TEST_FONT_PATH")
    if env_font:
        p = Path(env_font)
        if p.exists():
            return p
    candidates = [
        # macOS (system + user)
        "/System/Library/Fonts",
        "/Library/Fonts",
        str(Path.home() / "Library/Fonts"),
        # common Linux dirs
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        # Windows
        "C:/Windows/Fonts",
    ]
    exts = {".ttf", ".otf", ".ttc"}
    for root in candidates:
        p = Path(root)
        if not p.exists():
            continue
        for path in p.rglob("*"):
            if path.suffix.lower() in exts:
                return path
    return None


@pytest.mark.optional
def test_text_shape_generates_with_real_font():
    font_path = _find_any_system_font()
    if font_path is None:
        pytest.skip("no system font found in common locations")

    t = Text()
    g = t.generate(text="A", font=str(font_path), font_number=0, font_size=0.3, align="center")
    coords, offsets = g.as_arrays(copy=False)
    assert coords.size > 0 and offsets.size > 1
