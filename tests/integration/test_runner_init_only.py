from __future__ import annotations

import pytest


@pytest.mark.io
# What this tests (TEST_HARDENING_PLAN.md §Runner/設定)
# - run_sketch(init_only=True) returns before importing heavy deps (window creation).
def test_run_sketch_init_only_headless():
    """`init_only=True` なら重依存の import/Window 作成前に早期 return する。"""
    from api import run

    def draw(_t, _cc):  # not called in init_only path
        raise AssertionError("draw() must not be called in init_only path")

    # Should not raise, even without moderngl/pyglet available
    run(draw, canvas_size=(100, 100), render_scale=1, fps=1, use_midi=False, init_only=True)
