"""
Pipeline の miss/hit とキャッシュ条件の軽量ベンチ。

重依存を避けるため、効果は軽量な `rotate/scale/translate` のみを使用する。
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from api import E, G


@pytest.mark.perf
@pytest.mark.parametrize("mode", ["miss", "hit"])  # 1 ベンチ/1 呼び出し
def test_pipeline_miss_vs_hit(benchmark, mode: str):
    g = G.grid(nx=100, ny=100).scale(200, 200, 1)
    pipe = (
        E.pipeline.rotate(rotation=(0.0, 0.0, 17.0))
        .scale(scale=(1.05, 0.95, 1.0))
        .translate(delta=(1.0, 2.0, 0.0))
        .cache(maxsize=128)
        .build()
    )

    def miss():
        pipe.clear_cache()
        return pipe(g)

    def hit():
        return pipe(g)

    if mode == "hit":
        # 事前にキャッシュを温める
        pipe.clear_cache()
        _ = pipe(g)

    _ = benchmark(hit if mode == "hit" else miss)
    benchmark.extra_info.update(
        {
            "case": f"pipeline/miss_vs_hit/{mode}",
            "N": g.n_vertices,
            "M": g.n_lines,
        }
    )


## digest は廃止（Geometry 固有のダイジェストは持たない）。
## パイプラインのキャッシュキーは `lazy_signature_for(LazyGeometry)` に基づく。


@pytest.mark.perf
@pytest.mark.parametrize("cache", ["off", "on"])  # cache on/off 計測
def test_pipeline_cache_on_off(benchmark, cache: str):
    g = G.grid(nx=120, ny=120).scale(180, 180, 1)

    pipe_off = (
        E.pipeline.rotate(rotation=(0.0, 0.0, 15.0))
        .scale(scale=(1.02, 0.98, 1.0))
        .cache(maxsize=0)
        .build()
    )
    pipe_on = (
        E.pipeline.rotate(rotation=(0.0, 0.0, 15.0))
        .scale(scale=(1.02, 0.98, 1.0))
        .cache(maxsize=128)
        .build()
    )

    def nocache():
        return pipe_off(g)

    def cache_hit():
        return pipe_on(g)

    if cache == "on":
        # ヒットを安定させるために事前に 1 回計算
        _ = pipe_on(g)

    _ = benchmark(cache_hit if cache == "on" else nocache)
    benchmark.extra_info.update(
        {
            "case": f"pipeline/cache/{cache}",
            "N": g.n_vertices,
            "M": g.n_lines,
        }
    )
