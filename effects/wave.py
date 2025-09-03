from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry
from .registry import effect


@effect()
def wave(
    g: Geometry,
    *,
    amplitude: float = 0.1,
    frequency: float | tuple[float, float, float] = (0.1, 0.1, 0.1),
    phase: float = 0.0,
) -> Geometry:
    """座標値に基づくサイン波ゆらぎ（ウォブル）を各軸に適用する純関数エフェクト。

    Args:
        g: 入力ジオメトリ
        amplitude: 変位の大きさ
        frequency: 周波数（float なら全軸に同一値、タプルなら (fx, fy, fz)）
        phase: 位相オフセット（ラジアン）

    Returns:
        Geometry: ゆらぎが適用された新しいジオメトリ
    """
    coords, offsets = g.as_arrays(copy=False)
    if coords.size == 0:
        return Geometry(coords.copy(), offsets.copy())

    # frequency をタプルに正規化
    if isinstance(frequency, (int, float)):
        fx = fy = fz = float(frequency)
    else:
        if len(frequency) == 3:
            fx, fy, fz = float(frequency[0]), float(frequency[1]), float(frequency[2])
        else:
            fx = fy = fz = float(frequency[0]) if len(frequency) > 0 else 0.1

    out = coords.astype(np.float32, copy=True)
    two_pi = 2.0 * np.pi

    # 各軸にサイン波変位を適用（座標値ベース）
    if fx != 0.0:
        out[:, 0] += amplitude * np.sin(two_pi * fx * coords[:, 0] + phase)
    if fy != 0.0:
        out[:, 1] += amplitude * np.sin(two_pi * fy * coords[:, 1] + phase)
    if fz != 0.0 and out.shape[1] > 2:
        out[:, 2] += amplitude * np.sin(two_pi * fz * coords[:, 2] + phase)

    return Geometry(out, offsets.copy())

