from __future__ import annotations

import math

import numpy as np

from engine.core.geometry import Geometry
from engine.core.transform_utils import transform_combined


def test_transform_combined_order() -> None:
    # 単一点 (1,0,0) を対象に、Scale(2)→RotateZ(90°)→Translate(+1,0,0)
    g = Geometry.from_lines([np.array([[1.0, 0.0, 0.0]], dtype=np.float32)])
    out = transform_combined(
        g,
        center=(1.0, 0.0, 0.0),
        scale_factors=(2.0, 2.0, 2.0),
        rotate_angles=(0.0, 0.0, math.pi / 2),
    )
    # 期待: (1,2,0)
    assert np.allclose(out.coords, np.array([[1.0, 2.0, 0.0]], dtype=np.float32))
