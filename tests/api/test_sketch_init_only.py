from __future__ import annotations

from typing import Mapping

from api.sketch import run_sketch
from engine.core.geometry import Geometry


def _draw(_t: float, _cc: Mapping[int, float]) -> Geometry:
    # 最小ジオメトリ
    import numpy as np

    from engine.core.geometry import Geometry as G

    return G.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])


def test_run_sketch_init_only_returns_none_without_importing_heavy_deps() -> None:
    out = run_sketch(
        _draw, canvas_size=(100, 100), render_scale=1, fps=30, use_midi=False, init_only=True
    )
    assert out is None
