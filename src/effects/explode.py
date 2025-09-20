"""
explode エフェクト（中心からの放射発散）

- 重心から各頂点へ向かう方向ベクトルを正規化し、一定距離だけ外側へ移動させます。
- 形を「弾けさせる」効果で、線の位相は保持されます。

主なパラメータ:
- factor: 各頂点の移動距離 [mm]（0–50）。

注意:
- 全頂点が等距離で外側へ移動するため、連結は維持されますが交差が増える可能性があります。
"""

from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect


@effect()
def explode(g: Geometry, *, factor: float = 25.0) -> Geometry:
    """中心から外側へ頂点を放射状に移動させるエフェクト。

    引数:
        g: 入力ジオメトリ。
        factor: 移動距離（mm 単位）。

    返り値:
        変形後の `Geometry`。
    """
    coords, offsets = g.as_arrays(copy=False)
    if g.is_empty:
        return Geometry(coords.copy(), offsets.copy())

    center = coords.mean(axis=0)
    direction = coords - center
    # ゼロ長を避けつつ正規化
    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    safe = np.where(lengths > 1e-9, lengths, 1.0)
    unit = direction / safe
    amount = float(factor)
    out = coords + unit * amount
    return Geometry(out.astype(np.float32, copy=False), offsets.copy())


explode.__param_meta__ = {
    "factor": {"type": "number", "min": 0.0, "max": 50.0},
}
