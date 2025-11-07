from __future__ import annotations

import numpy as np
import pytest

from api import G
from common.param_utils import params_to_tuple
from engine.core.geometry import Geometry


def test_dynamic_dispatch_and_geometry() -> None:
    assert "sphere" in dir(G)
    g = G.sphere(subdivisions=0.0).realize()
    assert isinstance(g, Geometry)


def test_unknown_shape_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        getattr(G, "unknown_shape")()  # type: ignore[misc]


def test_params_to_tuple_stability() -> None:
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3)
    t1 = params_to_tuple({"arr": arr1, "s": {1, 2}, "b": b"abc"})
    t2 = params_to_tuple({"arr": arr2, "s": {2, 1}, "b": bytearray(b"abc")})
    assert t1 == t2


def test_shape_factory_lru_returns_same_instance() -> None:
    a = G.sphere(subdivisions=0.0)
    b = G.sphere(subdivisions=0.0)
    assert a.realize() is b.realize()


def test_from_lines_and_empty() -> None:
    lines = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)]
    g = G.from_lines(lines)
    assert isinstance(g, Geometry)
    assert not g.is_empty
    assert G.empty().is_empty
