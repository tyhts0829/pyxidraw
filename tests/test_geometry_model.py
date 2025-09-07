import numpy as np
import pytest

from engine.core.geometry import Geometry

# What this tests (TEST_PLAN.md §Geometry)
# - from_lines: 2D→3D 正規化、offsets 形状/末尾一致。
# - as_arrays(copy): 参照/コピーの挙動。
# - is_empty の判定と空表現。
# - translate の純関数性と digest の変化、digest 無効化時の例外。
# - 不正入力（shape(K,4)）で ValueError。


def test_from_lines_normalizes_2d_to_3d_and_offsets():
    lines = [
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 1.0]], dtype=np.float32),
    ]
    g = Geometry.from_lines(lines)

    # coords shape (N,3) with Z=0 filled
    assert g.coords.dtype == np.float32
    assert g.coords.ndim == 2 and g.coords.shape[1] == 3
    assert np.allclose(g.coords[:, 2], 0.0, rtol=1e-6, atol=1e-6)

    # offsets shape (M+1,) and last equals len(coords)
    assert g.offsets.dtype == np.int32
    assert g.offsets.ndim == 1
    assert g.offsets.shape[0] == 3  # two lines -> M+1 == 3
    assert int(g.offsets[-1]) == g.coords.shape[0]


def test_as_arrays_copy_semantics_and_is_empty():
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)])
    c0, o0 = g.as_arrays(copy=False)
    assert c0 is g.coords and o0 is g.offsets

    c1, o1 = g.as_arrays(copy=True)
    assert c1 is not g.coords and o1 is not g.offsets
    # mutate copies and ensure original is unchanged
    c1[0, 0] = 123.456
    assert not np.isclose(g.coords[0, 0], 123.456)

    empty = Geometry.from_lines([])
    assert empty.is_empty
    assert empty.coords.shape == (0, 3)
    assert np.array_equal(empty.offsets, np.array([0], dtype=np.int32))


def test_pure_functions_translate_and_digest_changes():
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])
    base_coords = g.coords.copy()

    # digest available by default
    d0 = g.digest
    assert isinstance(d0, (bytes, bytearray))

    g2 = g.translate(1.0, 2.0, 0.5)
    assert g2 is not g
    # original not mutated
    assert np.array_equal(g.coords, base_coords)
    # translated by delta
    delta = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    assert np.allclose(g2.coords, base_coords + delta, rtol=1e-6, atol=1e-6)
    # digest changes when content changes
    assert g2.digest != d0

    # zero-translate returns equal content but different instance
    g3 = g.translate(0.0, 0.0, 0.0)
    assert g3 is not g
    assert np.array_equal(g3.coords, g.coords)
    assert np.array_equal(g3.offsets, g.offsets)


def test_digest_disabled_via_env_raises(monkeypatch):
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])
    with pytest.raises(RuntimeError):
        _ = g.digest


def test_from_lines_invalid_input_raises_value_error():
    # shape (K, 4) is invalid
    bad = [np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)]
    with pytest.raises(ValueError):
        Geometry.from_lines(bad)
