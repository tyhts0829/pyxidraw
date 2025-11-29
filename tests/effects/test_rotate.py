from __future__ import annotations


import numpy as np

from effects.rotate import rotate
from engine.core.geometry import Geometry


def test_rotate_effect_around_origin() -> None:
    g = Geometry.from_lines([np.array([[1.0, 0.0, 0.0]], dtype=np.float32)])
    out = rotate(
        g,
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 90.0),
    )
    assert np.allclose(out.coords, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
