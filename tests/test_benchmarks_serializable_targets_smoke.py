import numpy as np

from engine.core.geometry import Geometry
from benchmarks.plugins.serializable_targets import SerializableEffectTarget


def _geom():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_translate_uses_delta_mapping():
    g = _geom()
    fx = SerializableEffectTarget("translate", {"translate": (1.0, 2.0, 3.0)})
    out = fx(g)
    c0, _ = g.as_arrays()
    c1, _ = out.as_arrays()
    assert np.allclose(c1, c0 + np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_noise_legacy_params_are_forwarded():
    g = _geom()
    # intensity/frequency are legacy names used by the benchmarks plugin
    fx = SerializableEffectTarget("noise", {"intensity": 0.0, "frequency": 0.0})
    out = fx(g)
    c0, _ = g.as_arrays()
    c1, _ = out.as_arrays()
    # with zero intensity, geometry should be unchanged
    assert np.allclose(c0, c1)

