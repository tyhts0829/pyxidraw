import numpy as np

from engine.core.geometry import Geometry
from effects.filling import fill
from effects.array import repeat


def test_filling_returns_geometry_and_adds_lines():
    # square polygon (closed)
    square = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]], dtype=np.float32)
    g = Geometry.from_lines([square])
    out = fill(g, mode="lines", density=0.2, angle_rad=0.0)
    assert isinstance(out, Geometry)
    c0, o0 = g.as_arrays()
    c1, o1 = out.as_arrays()
    assert len(o1) > len(o0)  # more lines than original
    assert c1.shape[0] >= c0.shape[0]


def test_array_duplicates_line_counts():
    base = Geometry.from_lines([
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32),
    ])
    dup = 0.3  # -> int(dup*10)=3 duplicates
    out = repeat(base, count=int(round(dup*10)), offset=(2, 0, 0), angles_rad_step=(np.pi, np.pi, np.pi), scale=(1.0, 1.0, 1.0), pivot=(0, 0, 0))
    c0, o0 = base.as_arrays()
    c1, o1 = out.as_arrays()
    n0 = len(o0) - 1
    n1 = len(o1) - 1
    assert n1 == n0 * (1 + int(dup * 10))
    assert c1.shape[0] >= c0.shape[0]
