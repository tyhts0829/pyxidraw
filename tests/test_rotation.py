"""
effects.rotate モジュールのテスト
"""
import math

import numpy as np
import pytest

from effects.rotate import rotate
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestRotate:
    def test_basic_rotation(self, simple_geometry):
        """基本的な回転テスト"""
        result = rotate(simple_geometry, angles_rad=(1.5707963267948966, 0.0, 0.0))
        assert isinstance(result, Geometry)
        assert len(result.coords) > 0
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_no_rotation(self, simple_geometry):
        """回転なしテスト"""
        result = rotate(simple_geometry, angles_rad=(0.0, 0.0, 0.0))
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_z_axis_rotation(self, simple_geometry):
        """Z軸回転テスト（90度）"""
        result = rotate(simple_geometry, angles_rad=(0.0, 0.0, 1.5707963267948966))  # 90度
        assert isinstance(result, Geometry)
        
        # 最初の点 [1, 0, 0] が約 [0, 1, 0] になることを確認
        # （Z軸90度回転で x→-y, y→x）
        original_point = simple_geometry.coords[1]  # [1, 0, 0]
        rotated_point = result.coords[1]
        expected = np.array([0, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(rotated_point, expected, atol=1e-6)

    def test_with_center(self, simple_geometry):
        """中心点指定回転テスト"""
        center = (0.5, 0.5, 0.0)
        result = rotate(simple_geometry, pivot=center, angles_rad=(0.0, 0.0, 1.5707963267948966))
        assert isinstance(result, Geometry)

    def test_multiple_axis_rotation(self, simple_geometry):
        """複数軸回転テスト"""
        result = rotate(simple_geometry, angles_rad=(0.1 * 2 * np.pi, 0.2 * 2 * np.pi, 0.3 * 2 * np.pi))
        assert isinstance(result, Geometry)
        assert len(result.coords) == len(simple_geometry.coords)

    def test_edge_case_full_rotation(self, simple_geometry):
        """1回転テスト（360度 = 1.0）"""
        result = rotate(simple_geometry, angles_rad=(0.0, 0.0, 2 * np.pi))
        assert isinstance(result, Geometry)
        # 1回転後は元の位置に戻る（浮動小数点の精度を考慮）
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-4, atol=1e-15)

    def test_negative_rotation(self, simple_geometry):
        """負の回転テスト"""
        result = rotate(simple_geometry, angles_rad=(0.0, 0.0, -1.5707963267948966))
        assert isinstance(result, Geometry)

    def test_large_rotation(self, simple_geometry):
        """大きな回転値テスト"""
        result = rotate(simple_geometry, angles_rad=(5.0 * 2 * np.pi, 3.0 * 2 * np.pi, 2.0 * 2 * np.pi))
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        result = rotate(simple_geometry, angles_rad=(0.1 * 2 * np.pi, 0.2 * 2 * np.pi, 0.3 * 2 * np.pi))
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)
