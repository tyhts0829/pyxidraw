"""共通フィクスチャ（Phase 1）。

- 乱数シード固定
- 小さな Geometry 試料
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from engine.core.geometry import Geometry


@pytest.fixture(scope="session", autouse=True)
def np_seed() -> None:
    """NumPy の乱数を固定。"""
    np.random.seed(12345)


@pytest.fixture()
def geom_empty() -> Geometry:
    return Geometry.from_lines([])


@pytest.fixture()
def geom_line2() -> Geometry:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([pts])


@pytest.fixture()
def geom_two_lines() -> Geometry:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([a, b])


@pytest.fixture()
def env_no_digest(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    yield
    monkeypatch.delenv("PXD_DISABLE_GEOMETRY_DIGEST", raising=False)
