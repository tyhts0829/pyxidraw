"""
ripple エフェクト（座標依存サイン波）

- 座標値に比例したサイン波変位を各軸に加え、周期的なうねりを与えます。
- `wobble` よりも整った波形で、周波数/位相の調整により規則的なリップル表現が可能です。

パラメータ:
- amplitude [mm], frequency [cycles/unit], phase [rad]。

注意:
- 単位は正規化値ではなく座標系の実寸（mm 相当）です。
"""

from __future__ import annotations

import numpy as np

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect


@effect()
def ripple(
    g: Geometry,
    *,
    amplitude: float = 1.5,
    frequency: float | Vec3 = 0.03,
    phase: float = 0.0,
) -> Geometry:
    """座標値に基づくサイン波ゆらぎ（ウォブル）を各軸に適用する純関数エフェクト。

    引数:
        g: 入力ジオメトリ
        amplitude: 変位の大きさ（座標単位, mm 相当）。0..1 正規化ではありません。
        frequency: 空間周波数 [cycles per unit]。float なら全軸に同一値、タプルなら (fx, fy, fz)。
        phase: 位相オフセット（ラジアン）

    返り値:
        Geometry: ゆらぎが適用された新しいジオメトリ
    """
    coords, offsets = g.as_arrays(copy=False)
    if g.is_empty:
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


ripple.__param_meta__ = {
    "amplitude": {"type": "number", "min": 0.0},
    "frequency": {"type": "vec3"},
    "phase": {"type": "number"},
}
