from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")  # noqa: F401 - 重依存の存在確認

from effects.displace import displace
from engine.core.geometry import Geometry


@pytest.mark.smoke
def test_displace_amplitude_zero_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    out = displace(g, amplitude_mm=0.0, spatial_freq=0.5, t_sec=0.0)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)


@pytest.mark.smoke
def test_displace_amplitude_vec_zero_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    out = displace(g, amplitude_mm=(0.0, 0.0, 0.0), spatial_freq=0.5, t_sec=0.0)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)


@pytest.mark.smoke
def test_displace_amplitude_vec_masks_axes() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    out = displace(g, amplitude_mm=(10.0, 0.0, 0.0), spatial_freq=0.5, t_sec=0.0)
    # y,z 成分は変化しない（x のみ揺れる）
    assert np.allclose(out.coords[:, 1:], g.coords[:, 1:])
    assert np.array_equal(out.offsets, g.offsets)


@pytest.mark.smoke
def test_displace_amplitude_gradient_with_zero_amplitude_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    out = displace(
        g,
        amplitude_mm=0.0,
        spatial_freq=0.5,
        amplitude_gradient=(1.0, -1.0, 0.5),
        t_sec=0.0,
    )
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)


@pytest.mark.smoke
def test_displace_amplitude_gradient_zero_keeps_axis_mask_behavior() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    out = displace(
        g,
        amplitude_mm=(10.0, 0.0, 0.0),
        spatial_freq=0.5,
        amplitude_gradient=0.0,
        t_sec=0.0,
    )
    # y,z 成分は変化しない（x のみ揺れる）
    assert np.allclose(out.coords[:, 1:], g.coords[:, 1:])
    assert np.array_equal(out.offsets, g.offsets)
