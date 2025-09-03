"""
effects.translation モジュールのテスト
"""
import numpy as np
import pytest

from effects.translation import translation
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestTranslation:
    def test_basic_translation(self, simple_geometry):
        """基本移動テスト"""
        result = translation(simple_geometry, offset_x=1.0, offset_y=2.0, offset_z=3.0)
        assert isinstance(result, Geometry)
        
        # 全ての点が指定量だけ移動することを確認
        expected_offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected_coords = simple_geometry.coords + expected_offset
        np.testing.assert_allclose(result.coords, expected_coords, rtol=1e-6)

    def test_no_translation(self, simple_geometry):
        """移動なしテスト"""
        result = translation(simple_geometry, offset_x=0.0, offset_y=0.0, offset_z=0.0)
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_x_only_translation(self, simple_geometry):
        """X軸のみ移動テスト"""
        result = translation(simple_geometry, offset_x=5.0, offset_y=0.0, offset_z=0.0)
        assert isinstance(result, Geometry)
        
        # X座標のみが変化することを確認
        np.testing.assert_allclose(result.coords[:, 0], simple_geometry.coords[:, 0] + 5.0, rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 1], simple_geometry.coords[:, 1], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 2], simple_geometry.coords[:, 2], rtol=1e-6)

    def test_y_only_translation(self, simple_geometry):
        """Y軸のみ移動テスト"""
        result = translation(simple_geometry, offset_x=0.0, offset_y=-3.0, offset_z=0.0)
        assert isinstance(result, Geometry)
        
        # Y座標のみが変化することを確認
        np.testing.assert_allclose(result.coords[:, 0], simple_geometry.coords[:, 0], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 1], simple_geometry.coords[:, 1] - 3.0, rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 2], simple_geometry.coords[:, 2], rtol=1e-6)

    def test_z_only_translation(self, simple_geometry):
        """Z軸のみ移動テスト"""
        result = translation(simple_geometry, offset_x=0.0, offset_y=0.0, offset_z=2.5)
        assert isinstance(result, Geometry)
        
        # Z座標のみが変化することを確認
        np.testing.assert_allclose(result.coords[:, 0], simple_geometry.coords[:, 0], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 1], simple_geometry.coords[:, 1], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 2], simple_geometry.coords[:, 2] + 2.5, rtol=1e-6)

    def test_negative_translation(self, simple_geometry):
        """負の移動テスト"""
        result = translation(simple_geometry, offset_x=-1.0, offset_y=-2.0, offset_z=-3.0)
        assert isinstance(result, Geometry)
        
        expected_offset = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        expected_coords = simple_geometry.coords + expected_offset
        np.testing.assert_allclose(result.coords, expected_coords, rtol=1e-6)

    def test_large_translation(self, simple_geometry):
        """大きな移動テスト"""
        result = translation(simple_geometry, offset_x=1000.0, offset_y=500.0, offset_z=-200.0)
        assert isinstance(result, Geometry)

    def test_small_translation(self, simple_geometry):
        """小さな移動テスト"""
        result = translation(simple_geometry, offset_x=0.001, offset_y=0.002, offset_z=0.003)
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        result = translation(simple_geometry, offset_x=10.0, offset_y=-5.0, offset_z=7.5)
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)

    def test_double_translation(self, simple_geometry):
        """二重移動テスト（移動の合成）"""
        # 最初の移動
        result1 = translation(simple_geometry, offset_x=1.0, offset_y=2.0, offset_z=3.0)
        # 二回目の移動
        result2 = translation(result1, offset_x=4.0, offset_y=5.0, offset_z=6.0)
        
        # 合計移動量と同じ結果になることを確認
        result_direct = translation(simple_geometry, offset_x=5.0, offset_y=7.0, offset_z=9.0)
        np.testing.assert_allclose(result2.coords, result_direct.coords, rtol=1e-6)
