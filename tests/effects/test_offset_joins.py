from __future__ import annotations

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")  # noqa: F401 - 依存確認のみ

from effects.offset import offset
from engine.core.geometry import Geometry


@pytest.mark.smoke
@pytest.mark.parametrize("join", ["round", "mitre", "bevel"])  # noqa: PT006
def test_offset_supports_three_join_styles(join: str) -> None:
    g = Geometry.from_lines(
        [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)]
    )
    out = offset(g, join=join, distance=1.0, segments_per_circle=8)
    assert isinstance(out, Geometry)
    assert len(out.offsets) >= 2
