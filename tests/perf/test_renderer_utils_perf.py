"""
Renderer 前処理（Geometry → VBO/IBO 変換）のベンチ。
GPU 依存はなく、純粋に配列生成コストのみを測る。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")
pytest.importorskip("moderngl")

from engine.core.geometry import Geometry
from engine.render.renderer import _geometry_to_vertices_indices


def _make_polyline(n: int) -> np.ndarray:
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    ys = np.sin(xs * np.pi).astype(np.float32)
    return np.stack([xs, ys, np.zeros_like(xs)], axis=1)


@pytest.mark.perf
def test_geometry_to_indices_small(benchmark):
    # 複数ラインの小規模ケース（合計 ~ 12_000 頂点）
    lines = [_make_polyline(2000), _make_polyline(2000), _make_polyline(2000)]
    g = Geometry.from_lines(lines)
    pri = 0xFFFFFFFF

    def target():
        return _geometry_to_vertices_indices(g, pri)

    _ = benchmark(target)
    benchmark.extra_info.update(
        {"case": "renderer/geom_to_indices_S", "N": g.n_vertices, "M": g.n_lines}
    )


@pytest.mark.perf
def test_geometry_to_indices_medium(benchmark):
    # 中規模ケース（合計 ~ 100_000 頂点）
    lines = [_make_polyline(20000) for _ in range(5)]
    g = Geometry.from_lines(lines)
    pri = 0xFFFFFFFF

    def target():
        return _geometry_to_vertices_indices(g, pri)

    _ = benchmark(target)
    benchmark.extra_info.update(
        {"case": "renderer/geom_to_indices_M", "N": g.n_vertices, "M": g.n_lines}
    )
