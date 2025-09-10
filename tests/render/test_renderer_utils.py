from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry
from engine.render.renderer import _geometry_to_vertices_indices
from util.constants import PRIMITIVE_RESTART_INDEX


def test_geometry_to_vertices_indices_inserts_primitive_restart() -> None:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([a, b])
    verts, inds = _geometry_to_vertices_indices(g, PRIMITIVE_RESTART_INDEX)
    # N = 5, lines=2 → indices=5+2
    assert len(verts) == 5
    assert len(inds) == 7
    # 1本目の後に PR が入る（位置=2）
    assert inds[2] == PRIMITIVE_RESTART_INDEX
    # 2本目の後に PR が入る（位置=2 + 1 + 3 = 6）
    assert inds[6] == PRIMITIVE_RESTART_INDEX
