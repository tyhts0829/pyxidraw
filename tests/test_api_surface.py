from __future__ import annotations

from tests._utils.dummies import install as install_dummies

# What this tests (TEST_PLAN.md §Pipeline/API/Smoke)
# - Public API surface can be imported after installing dummies for optional deps.
# - A minimal flow `G.polygon -> E.pipeline.rotate(...).build()(g)` returns Geometry.



def test_api_import_and_min_flow():
    # Install dummies for heavy optional deps before importing api/shapes/effects
    install_dummies()

    from api import E, G, Geometry, run  # noqa: F401

    g = G.polygon(n_sides=3)
    pipe = E.pipeline.rotate(pivot=(0.0, 0.0, 0.0), angles_rad=(0.0, 0.0, 0.1)).build()
    out = pipe(g)
    assert isinstance(out, Geometry)
