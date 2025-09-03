"""
effects.scaling モジュールのテスト
"""
import numpy as np
import pytest

from effects.scaling import scaling
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestScaling:
    def test_uniform_scaling(self, simple_geometry):
        """均一スケーリングテスト"""
        result = scaling(simple_geometry, scale=(2.0, 2.0, 2.0))
        assert isinstance(result, Geometry)
        
        # 元の点 [1, 0, 0] が [2, 0, 0] になることを確認
        original_point = simple_geometry.coords[1]  # [1, 0, 0]
        scaled_point = result.coords[1]
        expected = original_point * 2.0
        np.testing.assert_allclose(scaled_point, expected, rtol=1e-6)

    def test_no_scaling(self, simple_geometry):
        """スケーリングなしテスト（1.0倍）"""
        result = scaling(simple_geometry, scale=(1.0, 1.0, 1.0))
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_non_uniform_scaling(self, simple_geometry):
        """非均一スケーリングテスト"""
        result = scaling(simple_geometry, scale=(2.0, 3.0, 0.5))
        assert isinstance(result, Geometry)
        
        # 元の点 [1, 1, 0] の変換を確認
        original_point = simple_geometry.coords[2]  # [1, 1, 0]
        scaled_point = result.coords[2]
        expected = np.array([2.0, 3.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(scaled_point, expected, rtol=1e-6)

    def test_with_center(self, simple_geometry):
        """中心点指定スケーリングテスト"""
        center = (0.5, 0.5, 0.0)
        result = scaling(simple_geometry, center=center, scale=(2.0, 2.0, 2.0))
        assert isinstance(result, Geometry)

    def test_zero_scaling(self, simple_geometry):
        """ゼロスケーリングテスト"""
        result = scaling(simple_geometry, scale=(0.0, 1.0, 1.0))
        assert isinstance(result, Geometry)
        # X座標がすべて0になることを確認
        assert np.allclose(result.coords[:, 0], 0.0)

    def test_negative_scaling(self, simple_geometry):
        """負のスケーリングテスト（反転）"""
        result = scaling(simple_geometry, scale=(-1.0, 1.0, 1.0))
        assert isinstance(result, Geometry)
        
        # X座標が反転することを確認
        original_x = simple_geometry.coords[:, 0]
        scaled_x = result.coords[:, 0]
        np.testing.assert_allclose(scaled_x, -original_x, rtol=1e-6)

    def test_small_scaling(self, simple_geometry):
        """小さなスケーリングテスト"""
        result = scaling(simple_geometry, scale=(0.1, 0.1, 0.1))
        assert isinstance(result, Geometry)

    def test_large_scaling(self, simple_geometry):
        """大きなスケーリングテスト"""
        result = scaling(simple_geometry, scale=(100.0, 100.0, 100.0))
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        result = scaling(simple_geometry, scale=(1.5, 2.0, 0.8))
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)
