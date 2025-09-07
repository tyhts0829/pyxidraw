from __future__ import annotations

import pytest

from api import E, G

# What this tests (TEST_HARDENING_PLAN.md §E2E)
# - End-to-end digest snapshots for representative pipelines to catch broad regressions.


@pytest.mark.e2e
@pytest.mark.snapshot
def test_e2e_digest_polygon_fill(snapshot, digest_hex):
    g = G.polygon(n_sides=12)
    p = (
        E.pipeline.rotate(pivot=(0.0, 0.0, 0.0), angles_rad=(0.0, 0.0, 0.3))
        .fill(mode="lines", density=0.35)
        .build()
    )
    out = p(g)
    snapshot([digest_hex(out)], name="e2e_polygon_fill_v1")


@pytest.mark.e2e
@pytest.mark.snapshot
def test_e2e_digest_sphere_displace(snapshot, digest_hex):
    # Keep parameters modest to remain fast and deterministic
    g = G.sphere(subdivisions=0.4, sphere_type=0.3)
    p = E.pipeline.displace(amplitude_mm=0.1, spatial_freq=(0.02, 0.02, 0.02), t_sec=0.0).build()
    out = p(g)
    snapshot([digest_hex(out)], name="e2e_sphere_displace_v1")
