import numpy as np
import pytest

from api import E
from api.pipeline import to_spec, from_spec
from engine.core.geometry import Geometry


def _make_geom():
    lines = [
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestPipelineSerialization:
    def test_roundtrip(self):
        p = (
            E.pipeline
            .rotation(rotate=(0.0, 0.0, 0.25))
            .scaling(scale=(2.0, 2.0, 2.0))
            .translation(offset_x=5.0, offset_y=3.0)
            .build()
        )
        spec = to_spec(p)
        p2 = from_spec(spec)

        g = _make_geom()
        out1 = p(g)
        out2 = p2(g)
        c1, _ = out1.as_arrays()
        c2, _ = out2.as_arrays()
        np.testing.assert_allclose(c1, c2, rtol=1e-6)

    def test_invalid_name_raises(self):
        with pytest.raises(KeyError):
            from_spec([{"name": "__not_registered__", "params": {}}])

    def test_invalid_spec_types(self):
        with pytest.raises(TypeError):
            from_spec({"name": "rotation", "params": {}})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            from_spec([{"name": 123, "params": {}}])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            from_spec([{"name": "rotation", "params": 1}])  # type: ignore[list-item]

