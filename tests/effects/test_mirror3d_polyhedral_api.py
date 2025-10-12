from __future__ import annotations

import numpy as np

from effects.mirror3d import mirror3d
from engine.core.geometry import Geometry


def test_mirror3d_polyhedral_T_rotation_count() -> None:
    # 単点から T の回転群（12 元）を適用 → 12 本
    p = np.array([[0.3, 0.4, 0.5]], dtype=np.float32)
    g = Geometry.from_lines([p])
    out = mirror3d(g, mode="polyhedral", group="T", use_reflection=False)
    _, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 12


def test_mirror3d_polyhedral_T_with_reflections_count_24() -> None:
    # 反射込みで 24 本（T_d）
    p = np.array([[0.31, 0.41, 0.59]], dtype=np.float32)
    g = Geometry.from_lines([p])
    out = mirror3d(g, mode="polyhedral", group="T", use_reflection=True)
    _, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 24


def test_mirror3d_polyhedral_O_counts() -> None:
    p = np.array([[0.21, 0.31, 0.41]], dtype=np.float32)
    g = Geometry.from_lines([p])
    out = mirror3d(g, mode="polyhedral", group="O", use_reflection=False)
    _, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 24
    out = mirror3d(g, mode="polyhedral", group="O", use_reflection=True)
    _, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 48


def test_mirror3d_polyhedral_I_counts() -> None:
    p = np.array([[0.13, 0.21, 0.34]], dtype=np.float32)
    g = Geometry.from_lines([p])
    out = mirror3d(g, mode="polyhedral", group="I", use_reflection=False)
    _, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 60
    out = mirror3d(g, mode="polyhedral", group="I", use_reflection=True)
    _, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 120
