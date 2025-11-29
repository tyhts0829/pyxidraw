"""
rotate エフェクト（回転）

- 中心の選択（auto_center or pivot）に基づき、XYZ 各軸の回転角（ラジアン）を適用。
- 実装は `Geometry.rotate` に委譲し、新規インスタンスを返す純関数。

パラメータ:
- auto_center: True ならジオメトリの平均座標を中心に使用。False なら `pivot` を使用。
- pivot: 回転中心（Vec3）。`auto_center=False` のときのみ有効。
- angles_rad: (rx, ry, rz) [rad]（右手系）。
"""

from __future__ import annotations

import numpy as np

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect

PARAM_META = {
    "auto_center": {"type": "bool"},
    "pivot": {
        "type": "vec3",
        "min": (-300.0, -300.0, -300.0),
        "max": (300.0, 300.0, 300.0),
    },
    "angles_rad": {
        "type": "vec3",
        "min": (-np.pi, -np.pi, -np.pi),
        "max": (np.pi, np.pi, np.pi),
    },
}


@effect()
def rotate(
    g: Geometry,
    *,
    auto_center: bool = True,
    pivot: Vec3 = (0.0, 0.0, 0.0),
    angles_rad: Vec3 = (0, 0, 0),
) -> Geometry:
    """回転（auto_center 対応）。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。
    auto_center : bool, default True
        True なら平均座標を中心に使用。False なら `pivot` を使用。
    pivot : tuple[float, float, float], default (0.0,0.0,0.0)
        回転の中心（`auto_center=False` のとき有効）。
    angles_rad : tuple[float, float, float], default (π/4, π/4, π/4)
        (rx, ry, rz) ラジアン角。
    """
    # 角度を正規化
    rx, ry, rz = float(angles_rad[0]), float(angles_rad[1]), float(angles_rad[2])

    # 中心を決定（auto_center 優先）
    if auto_center:
        coords, offsets = g.as_arrays(copy=False)
        if coords.shape[0] == 0:
            # 空ジオメトリは no-op（Geometry.rotate と整合）
            return Geometry(coords.copy(), offsets.copy())
        center = tuple(coords.mean(axis=0).astype(np.float32))  # type: ignore[assignment]
    else:
        cx, cy, cz = pivot
        center = (cx, cy, cz)

    return g.rotate(x=rx, y=ry, z=rz, center=center)


rotate.__param_meta__ = PARAM_META
