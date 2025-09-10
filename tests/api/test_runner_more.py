from __future__ import annotations

import sys

import pytest

from api.runner import run_sketch


def _dummy_draw(t: float, cc):  # noqa: ANN001
    import numpy as np

    from engine.core.geometry import Geometry

    return Geometry.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])


def test_runner_midi_strict_true_exits_when_mido_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # mido を擬似的に “存在しない” 状態へ
    monkeypatch.setitem(sys.modules, "mido", None)
    # strict=True 明示、init_only=True で GL を避ける
    with pytest.raises(SystemExit) as ei:
        run_sketch(
            _dummy_draw,
            canvas_size=(100, 100),
            render_scale=1,
            fps=30,
            use_midi=True,
            midi_strict=True,
            init_only=True,
        )
    assert ei.value.code == 2


def test_runner_midi_non_strict_fallback_when_mido_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mido", None)
    # non-strict なら例外にならずに Null 実装へフォールバックし、その後 init_only=True で早期 return
    out = run_sketch(
        _dummy_draw,
        canvas_size=(100, 100),
        render_scale=1,
        fps=30,
        use_midi=True,
        midi_strict=False,
        init_only=True,
    )
    assert out is None


def test_runner_env_var_controls_midi_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    # midi_strict=None で env を参照
    monkeypatch.delenv("PYXIDRAW_MIDI_STRICT", raising=False)
    monkeypatch.setitem(sys.modules, "mido", None)
    monkeypatch.setenv("PYXIDRAW_MIDI_STRICT", "true")
    with pytest.raises(SystemExit) as ei:
        run_sketch(
            _dummy_draw,
            canvas_size=(100, 100),
            render_scale=1,
            fps=30,
            use_midi=True,
            midi_strict=None,
            init_only=True,
        )
    assert ei.value.code == 2


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
