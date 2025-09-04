from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry
from common.param_utils import norm_to_range
from .registry import effect


@effect()
def explode(g: Geometry, *, factor: float = 0.2) -> Geometry:
    """中心から外側へ頂点を放射状に移動させるエフェクト。

    Args:
        g: 入力ジオメトリ
        factor: 移動係数（0..1 相当のスケールを想定）

    Returns:
        Geometry: 変形後のジオメトリ
    """
    coords, offsets = g.as_arrays(copy=False)
    if coords.size == 0:
        return Geometry(coords.copy(), offsets.copy())

    center = coords.mean(axis=0)
    direction = coords - center
    # ゼロ長を避けつつ正規化
    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    safe = np.where(lengths > 1e-9, lengths, 1.0)
    unit = direction / safe
    # 0..1 → mm スケールへ写像（線形）。等価: factor*MAX_OFFSET
    MAX_OFFSET = 50.0
    amount = norm_to_range(float(factor), 0.0, MAX_OFFSET)
    out = coords + unit * amount
    return Geometry(out.astype(np.float32, copy=False), offsets.copy())

explode.__param_meta__ = {
    "factor": {"type": "number", "min": 0.0, "max": 1.0},
}
