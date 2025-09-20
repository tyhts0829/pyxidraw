"""
Geometry 周辺のマイクロ/スモールベンチ。

計測は pytest-benchmark の `benchmark` フィクスチャを用いる。
メモリは `tests/perf/_mem.py` のユーティリティで補助的に観測する。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from engine.core.geometry import Geometry

from ._mem import measure_memory


@pytest.mark.perf
def test_from_lines_small(benchmark):
    # S 規模（約 10_000 頂点）
    xs = np.linspace(0, 1, 5_000, dtype=np.float32)
    ys = np.sin(xs * np.pi).astype(np.float32)
    line = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
    lines = [line, line[::-1]]

    def target():
        g = Geometry.from_lines(lines)
        _ = g.n_vertices, g.n_lines  # 軽い利用
        return g

    g, mem = measure_memory(target)
    benchmark.extra_info.update(
        {"case": "geometry/from_lines_S", "N": g.n_vertices, "M": g.n_lines, **mem}
    )
    benchmark.pedantic(lambda: Geometry.from_lines(lines), rounds=10, warmup_rounds=2)


@pytest.mark.perf
def test_from_lines_medium(benchmark):
    # M 規模（約 100_000 頂点）
    n = 50_000
    xs = np.linspace(0, 1, n, dtype=np.float32)
    ys = np.sin(xs * np.pi).astype(np.float32)
    line = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
    lines = [line, line[::-1]]

    def target():
        g = Geometry.from_lines(lines)
        _ = g.n_vertices, g.n_lines  # 軽い利用
        return g

    g, mem = measure_memory(target)
    benchmark.extra_info.update(
        {
            "case": "geometry/from_lines_M",
            "N": g.n_vertices,
            "M": g.n_lines,
            **mem,
        }
    )
    benchmark.pedantic(lambda: Geometry.from_lines(lines), rounds=5, warmup_rounds=1)


@pytest.mark.perf
@pytest.mark.parametrize("op", ["rotate", "scale", "concat"])  # 1 ベンチ/1 呼び出しに分離
def test_rotate_scale_concat_micro(benchmark, op: str):
    # 入力生成（N ~ 20_000）
    n = 10_000
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    line = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    g = Geometry.from_lines([line, line[::-1]])

    def do_rotate():
        return g.rotate(z=0.5)

    def do_scale():
        return g.scale(1.1, 0.9, 1.0)

    def do_concat():
        return g.concat(g)

    fn = {"rotate": do_rotate, "scale": do_scale, "concat": do_concat}[op]
    # 事前に 1 回だけメモリ測定（ベンチ結果とは独立）
    _, mem = measure_memory(fn)
    _ = benchmark(fn)
    benchmark.extra_info.update(
        {
            "case": f"geometry/micro_ops/{op}",
            "N": g.n_vertices,
            "M": g.n_lines,
            **mem,
        }
    )
