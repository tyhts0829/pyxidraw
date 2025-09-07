"""
translate エフェクト（平行移動）

- 入力を XYZ のオフセットだけ移動する純関数です。
- オフセットがゼロ、または空ジオメトリの場合はコピーを返す高速経路があります。

パラメータ:
- delta: (dx, dy, dz) [mm]。
"""

from __future__ import annotations

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect


@effect()
def translate(
    g: Geometry,
    *,
    delta: Vec3 = (0.0, 0.0, 0.0),
) -> Geometry:
    """指定ベクトルで平行移動（Geometry メソッドに委譲）。"""
    ox, oy, oz = float(delta[0]), float(delta[1]), float(delta[2])
    if ox == 0.0 and oy == 0.0 and oz == 0.0 or g.is_empty:
        coords, offsets = g.as_arrays(copy=False)
        return Geometry(coords.copy(), offsets.copy())
    return g.translate(ox, oy, oz)


translate.__param_meta__ = {
    "delta": {"type": "vec3"},
}

# 後方互換クラスは廃止（関数APIのみ）
