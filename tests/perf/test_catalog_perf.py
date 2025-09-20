"""
全 shape / 全 effect の網羅的なマイクロベンチ。

方針:
- shape: レジストリ登録済みのすべての shape を既定引数で実行し、生成時間を計測。
- effect: 代表ジオメトリに対して、登録済み effect を既定引数で適用し、実行時間を計測。

注意:
- オプショナル依存（例: shapely）を要求する effect は、環境に無い場合 ImportError 等で
  失敗する可能性がある。その場合は該当 effect を skip する（テスト失敗にはしない）。
- 1 テスト 1 ベンチの原則に従い、parametrize で分割する。
"""

from __future__ import annotations

import importlib
from typing import List

import pytest

pytest.importorskip("pytest_benchmark")

from api import G
from effects.registry import get_effect, list_effects
from shapes.registry import list_shapes

from ._mem import measure_memory

# 代表入力（effects 用）: 中規模グリッド
_BASE_GEOM = G.grid(subdivisions=(100.0, 100.0)).scale(200.0, 200.0, 1.0)


def _try_import_all_effect_modules() -> None:
    """effects パッケージ配下のモジュールをできる範囲で import。

    effects/__init__ は重依存を含むため、環境によって ImportError が起きうる。
    ここでは安全のため、個別モジュールに対して best-effort で import を試み、
    失敗したものは黙ってスキップする。
    """

    try:
        importlib.import_module("effects")  # まとめて登録（成功すれば十分）
        return
    except Exception:
        pass

    # フォールバック: 個別に試す（最低限コアな rotate/scale/translate は期待）
    candidates = [
        "affine",
        "boldify",
        "collapse",
        "dash",
        "displace",
        "explode",
        "extrude",
        "fill",
        "offset",
        "repeat",
        "ripple",
        "rotate",
        "scale",
        "subdivide",
        "translate",
        "trim",
        "twist",
        "weave",
        "wobble",
    ]
    for name in candidates:
        try:
            importlib.import_module(f"effects.{name}")
        except Exception:
            continue


_try_import_all_effect_modules()


def _param_ids(names: List[str]) -> List[str]:
    # 表示短縮のためそのまま
    return names


# ---- Shapes -----------------------------------------------------------------
_SHAPES = list_shapes()


@pytest.mark.perf
@pytest.mark.parametrize("shape_name", _SHAPES, ids=_param_ids(_SHAPES))
def test_each_shape_default(benchmark, shape_name: str):
    def target():
        fn = getattr(G, shape_name)
        return fn()

    g, mem = measure_memory(target)
    _ = benchmark(lambda: getattr(G, shape_name)())
    benchmark.extra_info.update(
        {
            "case": f"shape/{shape_name}",
            "N": g.n_vertices,
            "M": g.n_lines,
            **mem,
        }
    )


# ---- Effects ----------------------------------------------------------------
_EFFECTS = list_effects()


@pytest.mark.perf
@pytest.mark.parametrize("effect_name", _EFFECTS, ids=_param_ids(_EFFECTS))
def test_each_effect_default(benchmark, effect_name: str):
    # effect(g, **defaults) を直接呼んで計測（Pipeline 経由のオーバーヘッドを除外）
    fn = get_effect(effect_name)

    def target():
        return fn(_BASE_GEOM)

    try:
        g, mem = measure_memory(target)
    except Exception as e:  # オプショナル依存不足や未対応入力は skip
        pytest.skip(f"effect '{effect_name}' skipped: {e}")
        return

    _ = benchmark(lambda: fn(_BASE_GEOM))
    benchmark.extra_info.update(
        {
            "case": f"effect/{effect_name}",
            "N": g.n_vertices,
            "M": g.n_lines,
            **mem,
        }
    )
