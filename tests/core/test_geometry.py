from __future__ import annotations

import numpy as np
import pytest

from engine.core.geometry import Geometry


def test_from_lines_normalizes_2d_and_offsets() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    g = Geometry.from_lines([xy])
    assert g.coords.shape == (3, 3)
    assert g.offsets.tolist() == [0, 3]
    assert np.allclose(g.coords[:, 2], 0.0)


def test_from_lines_1d_invalid_raises() -> None:
    bad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # 4 は 3 の倍数でない
    with pytest.raises(ValueError):
        Geometry.from_lines([bad])


def test_empty_geometry_properties() -> None:
    g = Geometry.from_lines([])
    assert g.is_empty
    assert g.coords.shape == (0, 3)
    assert g.offsets.tolist() == [0]


def test_translate_is_pure_and_new_instance() -> None:
    g0 = Geometry.from_lines([np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)])
    g1 = g0.translate(1.0, 2.0, 3.0)
    assert g1 is not g0
    assert np.allclose(g1.coords, g0.coords + np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(g1.offsets, g0.offsets)


def test_digest_disabled_raises(env_no_digest: None) -> None:
    g = Geometry.from_lines([np.array([[0, 0, 0]], dtype=np.float32)])
    with pytest.raises(RuntimeError):
        _ = g.digest


def test_digest_enabled_returns_cached_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PXD_DISABLE_GEOMETRY_DIGEST", raising=False)
    g2 = Geometry.from_lines([np.array([[0, 0, 0]], dtype=np.float32)])
    d1 = g2.digest
    d2 = g2.digest
    assert isinstance(d1, (bytes, bytearray)) and len(d1) == 16
    assert d1 is d2
