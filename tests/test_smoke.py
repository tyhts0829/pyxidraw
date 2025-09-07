from __future__ import annotations

import pytest

from tests._utils.dummies import install as install_dummies

# What this tests (TEST_PLAN.md §Smoke)
# - main.draw(t, cc) returns a Geometry with minimal CC mapping after installing dummies.


@pytest.mark.smoke
def test_main_draw_returns_geometry():
    install_dummies()
    import main
    from engine.core.geometry import Geometry

    # Minimal CC values used by main.draw; ensure keys exist
    cc = {i: 0.0 for i in range(0, 16)}
    # Set some reasonable non-zero factors for visibility but keep fast
    cc[1] = 0.1  # subdivisions
    cc[2] = 0.0  # sphere_type
    cc[8] = 0.2  # scale factor
    # other keys default to 0.0
    g = main.draw(t=0.0, cc=cc)
    assert isinstance(g, Geometry)
