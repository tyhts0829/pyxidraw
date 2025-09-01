"""
effects.rotation モジュールのテスト
"""
import math

import numpy as np
import pytest

from effects.rotation import Rotation
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestRotation:
    def test_init(self):
        """Rotationクラスの初期化テスト"""
        effect = Rotation()
        assert effect is not None

    def test_basic_rotation(self, simple_geometry):
        """基本的な回転テスト"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.25, 0.0, 0.0))
        assert isinstance(result, Geometry)
        assert len(result.coords) > 0
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_no_rotation(self, simple_geometry):
        """回転なしテスト"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.0, 0.0, 0.0))
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_z_axis_rotation(self, simple_geometry):
        """Z軸回転テスト（90度）"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.0, 0.0, 0.25))  # 0.25 * 2π = π/2 = 90度
        assert isinstance(result, Geometry)
        
        # 最初の点 [1, 0, 0] が約 [0, 1, 0] になることを確認
        # （Z軸90度回転で x→-y, y→x）
        original_point = simple_geometry.coords[1]  # [1, 0, 0]
        rotated_point = result.coords[1]
        expected = np.array([0, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(rotated_point, expected, atol=1e-6)

    def test_with_center(self, simple_geometry):
        """中心点指定回転テスト"""
        effect = Rotation()
        center = (0.5, 0.5, 0.0)
        result = effect(simple_geometry, center=center, rotate=(0.0, 0.0, 0.25))
        assert isinstance(result, Geometry)

    def test_multiple_axis_rotation(self, simple_geometry):
        """複数軸回転テスト"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.1, 0.2, 0.3))
        assert isinstance(result, Geometry)
        assert len(result.coords) == len(simple_geometry.coords)

    def test_edge_case_full_rotation(self, simple_geometry):
        """1回転テスト（360度 = 1.0）"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.0, 0.0, 1.0))
        assert isinstance(result, Geometry)
        # 1回転後は元の位置に戻る（浮動小数点の精度を考慮）
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-4, atol=1e-15)

    def test_negative_rotation(self, simple_geometry):
        """負の回転テスト"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.0, 0.0, -0.25))
        assert isinstance(result, Geometry)

    def test_large_rotation(self, simple_geometry):
        """大きな回転値テスト"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(5.0, 3.0, 2.0))
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        effect = Rotation()
        result = effect(simple_geometry, rotate=(0.1, 0.2, 0.3))
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)