import numpy as np
import pytest

pytest.importorskip("numba")

from shapes.torus import Torus

# What this tests (TEST_HARDENING_PLAN.md §Optional)
# - With real numba installed, a JIT-backed shape (Torus) generates valid geometry quickly.


@pytest.mark.optional
def test_numba_backed_torus_small_params():
    # keep segment counts small to minimize compile/runtime in CI
    g = Torus().generate(major_radius=0.2, minor_radius=0.08, major_segments=8, minor_segments=6)
    coords, offsets = g.as_arrays(copy=False)
    assert coords.dtype == np.float32
    assert coords.shape[1] == 3
    assert offsets.ndim == 1 and offsets.size > 1
