import numpy as np
import pytest

shapely = pytest.importorskip("shapely")

from effects.offset import offset as effect_offset
from engine.core.geometry import Geometry

# What this tests (TEST_HARDENING_PLAN.md §Optional)
# - With real shapely installed, effects.offset runs and returns non-empty geometry.


@pytest.mark.optional
def test_shapely_offset_applies_without_import_error():
    # simple square polyline (closed)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, len(coords)], dtype=np.int32)
    g = Geometry(coords, offsets)

    # run with explicit mm distance to avoid mapping variance
    g2 = effect_offset(g, distance_mm=5.0, join="round", segments_per_circle=8)

    c2, o2 = g2.as_arrays(copy=False)
    assert c2.size > 0 and o2.size > 1
    # expect geometry to be different from input
    assert not np.array_equal(c2, coords)
