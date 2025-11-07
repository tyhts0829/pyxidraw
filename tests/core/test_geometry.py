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


def test_constructor_normalizes_dtype_and_contiguity() -> None:
    coords = np.asfortranarray(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64))
    offsets = np.array([0, 2], dtype=np.int64)
    g = Geometry(coords, offsets)
    assert g.coords.dtype == np.float32
    assert g.offsets.dtype == np.int32
    assert g.coords.flags.c_contiguous is True
    assert g.offsets.flags.c_contiguous is True
    assert np.allclose(g.coords, coords)


def test_constructor_invalid_coords_shape_raises() -> None:
    coords = np.zeros((2, 2), dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    with pytest.raises(ValueError):
        Geometry(coords, offsets)


def test_constructor_invalid_offsets_raises() -> None:
    coords = np.zeros((2, 3), dtype=np.float32)
    bad_offsets = np.array([0, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        Geometry(coords, bad_offsets)


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


def test_len_and_count_properties() -> None:
    # 2 本の線（2D と 1D 混在）
    l2d = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # 2 点
    l1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)  # 2 点（1D → 3D 整形）
    g = Geometry.from_lines([l2d, l1d])
    assert len(g) == 2
    assert g.n_lines == 2
    assert g.n_vertices == 4


def test_concat_merges_coords_and_offsets() -> None:
    g1 = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])
    g2 = Geometry.from_lines([np.array([[2.0, 0.0, 0.0]], dtype=np.float32)])
    g = g1.concat(g2)

    # coords は縦結合され、offsets は後段がシフトされる
    expect_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    expect_offsets = np.array([0, 2, 3], dtype=np.int32)
    assert np.allclose(g.coords, expect_coords)
    assert np.array_equal(g.offsets, expect_offsets)
