from __future__ import annotations

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")  # noqa: F401 - 重依存の存在確認

from effects.offset import offset
from engine.core.geometry import Geometry


@pytest.mark.smoke
def test_offset_distance_zero_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])
    out = offset(g, distance=0.0)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)
