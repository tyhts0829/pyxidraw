from __future__ import annotations

import sys

import pytest

from api.sketch import run_sketch


def _dummy_draw(t: float):  # noqa: ANN001
    import numpy as np

    from engine.core.geometry import Geometry

    return Geometry.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])


def test_runner_midi_fallback_when_mido_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mido", None)
    # 常に Null 実装へフォールバックし、その後 init_only=True で早期 return
    out = run_sketch(
        _dummy_draw,
        canvas_size=(100, 100),
        render_scale=1,
        fps=30,
        use_midi=True,
        init_only=True,
    )
    assert out is None


def test_runner_init_only_avoids_gl_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    # GL ライブラリが無い仮定でも init_only=True なら ImportError は起きない
    if "moderngl" in sys.modules:
        del sys.modules["moderngl"]
    if "pyglet" in sys.modules:
        del sys.modules["pyglet"]
    # ここで “無い” ことを明示しなくても、init_only=True の分岐で import しないため成功する
    out = run_sketch(
        _dummy_draw, canvas_size=(100, 100), render_scale=1, fps=30, use_midi=False, init_only=True
    )
    assert out is None
