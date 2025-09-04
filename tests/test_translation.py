"""
effects.translation モジュールのテスト
"""
import numpy as np
import pytest

from effects.translation import translate
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestTranslate:
    def test_basic_translation(self, simple_geometry):
        """基本移動テスト"""
        result = translate(simple_geometry, delta=(1.0, 2.0, 3.0))
        assert isinstance(result, Geometry)
        
        # 全ての点が指定量だけ移動することを確認
        expected_offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected_coords = simple_geometry.coords + expected_offset
        np.testing.assert_allclose(result.coords, expected_coords, rtol=1e-6)

    def test_no_translation(self, simple_geometry):
        """移動なしテスト"""
        result = translate(simple_geometry, delta=(0.0, 0.0, 0.0))
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_x_only_translation(self, simple_geometry):
        """X軸のみ移動テスト"""
        result = translate(simple_geometry, delta=(5.0, 0.0, 0.0))
        assert isinstance(result, Geometry)
        
        # X座標のみが変化することを確認
        np.testing.assert_allclose(result.coords[:, 0], simple_geometry.coords[:, 0] + 5.0, rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 1], simple_geometry.coords[:, 1], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 2], simple_geometry.coords[:, 2], rtol=1e-6)

    def test_y_only_translation(self, simple_geometry):
        """Y軸のみ移動テスト"""
        result = translate(simple_geometry, delta=(0.0, -3.0, 0.0))
        assert isinstance(result, Geometry)
        
        # Y座標のみが変化することを確認
        np.testing.assert_allclose(result.coords[:, 0], simple_geometry.coords[:, 0], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 1], simple_geometry.coords[:, 1] - 3.0, rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 2], simple_geometry.coords[:, 2], rtol=1e-6)

    def test_z_only_translation(self, simple_geometry):
        """Z軸のみ移動テスト"""
        result = translate(simple_geometry, delta=(0.0, 0.0, 2.5))
        assert isinstance(result, Geometry)
        
        # Z座標のみが変化することを確認
        np.testing.assert_allclose(result.coords[:, 0], simple_geometry.coords[:, 0], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 1], simple_geometry.coords[:, 1], rtol=1e-6)
        np.testing.assert_allclose(result.coords[:, 2], simple_geometry.coords[:, 2] + 2.5, rtol=1e-6)

    def test_negative_translation(self, simple_geometry):
        """負の移動テスト"""
        result = translate(simple_geometry, delta=(-1.0, -2.0, -3.0))
        assert isinstance(result, Geometry)
        
        expected_offset = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        expected_coords = simple_geometry.coords + expected_offset
        np.testing.assert_allclose(result.coords, expected_coords, rtol=1e-6)

    def test_large_translation(self, simple_geometry):
        """大きな移動テスト"""
        result = translate(simple_geometry, delta=(1000.0, 500.0, -200.0))
        assert isinstance(result, Geometry)

    def test_small_translation(self, simple_geometry):
        """小さな移動テスト"""
        result = translate(simple_geometry, delta=(0.001, 0.002, 0.003))
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        result = translate(simple_geometry, delta=(10.0, -5.0, 7.5))
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)

    def test_double_translation(self, simple_geometry):
        """二重移動テスト（移動の合成）"""
        # 最初の移動
        result1 = translate(simple_geometry, delta=(1.0, 2.0, 3.0))
        # 二回目の移動
        result2 = translate(result1, delta=(4.0, 5.0, 6.0))
        
        # 合計移動量と同じ結果になることを確認
        result_direct = translate(simple_geometry, delta=(5.0, 7.0, 9.0))
        np.testing.assert_allclose(result2.coords, result_direct.coords, rtol=1e-6)
