from __future__ import annotations

import numpy as np

from tests._utils.dummies import install as install_dummies

# What this tests (TEST_PLAN.md §Pipeline/Equivalence)
# - Direct application of effects vs. Pipeline-built chain produce equal Geometry.


def test_pipeline_equals_direct_effects_application():
    install_dummies()
    from api import E
    from effects.rotate import rotate as fx_rotate
    from effects.translate import translate as fx_translate
    from engine.core.geometry import Geometry

    # Simple square polyline
    square = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    g = Geometry.from_lines([square])

    # Direct effects
    g1 = fx_translate(g, delta=(2.0, 0.0, 0.0))
    g1 = fx_rotate(g1, pivot=(0.0, 0.0, 0.0), angles_rad=(0.0, 0.0, 0.25))

    # Pipeline
    p = (
        E.pipeline.translate(delta=(2.0, 0.0, 0.0))
        .rotate(pivot=(0.0, 0.0, 0.0), angles_rad=(0.0, 0.0, 0.25))
        .build()
    )
    g2 = p(g)

    c1, o1 = g1.as_arrays(copy=False)
    c2, o2 = g2.as_arrays(copy=False)
    assert np.allclose(c1, c2, rtol=1e-6, atol=1e-6)
    assert np.array_equal(o1, o2)
