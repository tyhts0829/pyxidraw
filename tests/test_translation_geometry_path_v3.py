import numpy as np

from engine.core.geometry import Geometry
from effects.translate import translate


def test_translate_effect_geometry_path_returns_geometry():
    base = Geometry.from_lines([np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)])
    out = translate(base, delta=(5, -1, 2))
    assert isinstance(out, Geometry)
    c, o = out.as_arrays()
    assert o.tolist() == [0, 2]
    assert np.allclose(c[0], [5, -1, 2])
