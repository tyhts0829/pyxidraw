"""
scale エフェクト（拡大縮小）

- 中心の選択（auto_center or pivot）に基づき、各軸の倍率でスケール変換（非等方可）。
- 実装は `Geometry.scale` に委譲する純関数。

パラメータ:
- auto_center: True ならジオメトリの平均座標を中心に使用。False なら `pivot` を使用。
- pivot: 変換の中心（`auto_center=False` のとき有効）。
- scale: (sx, sy, sz) 倍率。
"""

from __future__ import annotations

import numpy as np

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect


@effect()
def scale(
    g: Geometry,
    *,
    auto_center: bool = True,
    pivot: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (0.75, 0.75, 0.75),
) -> Geometry:
    """スケール変換を適用（auto_center 対応）。

    引数:
        auto_center: True なら平均座標を中心に使用。False なら `pivot` を中心に使用
        pivot: スケーリングの中心（`auto_center=False` のとき有効）
        scale: 各軸の倍率（`(sx, sy, sz)`）

    返り値:
        スケーリング後の `Geometry`。
    """
    sx, sy, sz = scale

    # 中心を決定（auto_center 優先）
    if auto_center:
        coords, offsets = g.as_arrays(copy=False)
        if coords.shape[0] == 0:
            return Geometry(coords.copy(), offsets.copy())
        center = tuple(coords.mean(axis=0).astype(np.float32))  # type: ignore[assignment]
    else:
        cx, cy, cz = pivot
        center = (cx, cy, cz)

    return g.scale(sx, sy, sz, center=center)


# 後方互換クラスは廃止（関数APIのみ）
scale.__param_meta__ = {
    "auto_center": {"type": "bool"},
    "pivot": {
        "type": "vec3",
        "min": (-300.0, -300.0, -300.0),
        "max": (300.0, 300.0, 300.0),
    },
    "scale": {"type": "vec3", "min": (0.1, 0.1, 0.1), "max": (5.0, 5.0, 5.0)},
}
