"""
effects.array モジュールのテスト
"""
import numpy as np
import pytest

from effects.array import array
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestArray:
    def test_no_duplicates(self, simple_geometry):
        """複製なしテスト（n_duplicates=0.0）"""
        result = array(simple_geometry, n_duplicates=0.0)
        assert isinstance(result, Geometry)
        # 複製なしなので元と同じかそれに近い
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_basic_array(self, simple_geometry):
        """基本的な配列テスト"""
        result = array(simple_geometry, n_duplicates=0.5, offset=(10, 0, 0))
        assert isinstance(result, Geometry)
        # 複製により頂点数が増加
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_max_duplicates(self, simple_geometry):
        """最大複製テスト（n_duplicates=1.0）"""
        result = array(simple_geometry, n_duplicates=1.0, offset=(5, 5, 0))
        assert isinstance(result, Geometry)

    def test_x_offset_only(self, simple_geometry):
        """X軸オフセットのみテスト"""
        result = array(simple_geometry, n_duplicates=0.3, offset=(10, 0, 0))
        assert isinstance(result, Geometry)

    def test_y_offset_only(self, simple_geometry):
        """Y軸オフセットのみテスト"""
        result = array(simple_geometry, n_duplicates=0.3, offset=(0, 10, 0))
        assert isinstance(result, Geometry)

    def test_z_offset_only(self, simple_geometry):
        """Z軸オフセットのみテスト"""
        result = array(simple_geometry, n_duplicates=0.3, offset=(0, 0, 10))
        assert isinstance(result, Geometry)

    def test_diagonal_offset(self, simple_geometry):
        """対角線オフセットテスト"""
        result = array(simple_geometry, n_duplicates=0.4, offset=(5, 5, 5))
        assert isinstance(result, Geometry)

    def test_with_rotation(self, simple_geometry):
        """回転付き配列テスト"""
        result = array(
            simple_geometry,
            n_duplicates=0.5,
            offset=(10, 0, 0),
            rotate=(0.1, 0.0, 0.0)
        )
        assert isinstance(result, Geometry)

    def test_with_scaling(self, simple_geometry):
        """スケーリング付き配列テスト"""
        result = array(
            simple_geometry,
            n_duplicates=0.5,
            offset=(10, 0, 0),
            scale=(0.3, 0.3, 0.3)
        )
        assert isinstance(result, Geometry)

    def test_with_center(self, simple_geometry):
        """中心点指定配列テスト"""
        result = array(
            simple_geometry,
            n_duplicates=0.5,
            offset=(10, 0, 0),
            center=(5, 5, 0)
        )
        assert isinstance(result, Geometry)

    def test_combined_transformations(self, simple_geometry):
        """複合変換配列テスト"""
        result = array(
            simple_geometry,
            n_duplicates=0.6,
            offset=(8, 3, 2),
            rotate=(0.2, 0.1, 0.3),
            scale=(0.4, 0.6, 0.8),
            center=(1, 1, 1)
        )
        assert isinstance(result, Geometry)

    def test_negative_offset(self, simple_geometry):
        """負のオフセットテスト"""
        result = array(simple_geometry, n_duplicates=0.3, offset=(-5, -3, -2))
        assert isinstance(result, Geometry)

    def test_zero_offset(self, simple_geometry):
        """ゼロオフセットテスト"""
        result = array(simple_geometry, n_duplicates=0.3, offset=(0, 0, 0))
        assert isinstance(result, Geometry)

    def test_large_offset(self, simple_geometry):
        """大きなオフセットテスト"""
        result = array(simple_geometry, n_duplicates=0.3, offset=(100, 200, 50))
        assert isinstance(result, Geometry)

    def test_various_duplicate_counts(self, simple_geometry):
        """様々な複製数テスト"""
        for n_duplicates in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = array(simple_geometry, n_duplicates=n_duplicates, offset=(5, 0, 0))
            assert isinstance(result, Geometry)

    def test_rotation_variations(self, simple_geometry):
        """回転バリエーションテスト"""
        # X軸回転
        result_x = array(simple_geometry, n_duplicates=0.3, offset=(5, 0, 0), rotate=(0.2, 0.0, 0.0))
        # Y軸回転
        result_y = array(simple_geometry, n_duplicates=0.3, offset=(5, 0, 0), rotate=(0.0, 0.2, 0.0))
        # Z軸回転
        result_z = array(simple_geometry, n_duplicates=0.3, offset=(5, 0, 0), rotate=(0.0, 0.0, 0.2))
        
        assert isinstance(result_x, Geometry)
        assert isinstance(result_y, Geometry)
        assert isinstance(result_z, Geometry)

    def test_scale_variations(self, simple_geometry):
        """スケールバリエーションテスト"""
        # 拡大
        result_expand = array(simple_geometry, n_duplicates=0.3, offset=(5, 0, 0), scale=(0.8, 0.8, 0.8))
        # 縮小
        result_shrink = array(simple_geometry, n_duplicates=0.3, offset=(5, 0, 0), scale=(0.2, 0.2, 0.2))
        
        assert isinstance(result_expand, Geometry)
        assert isinstance(result_shrink, Geometry)

    def test_preserves_original_structure(self, simple_geometry):
        """元構造保持テスト"""
        result = array(simple_geometry, n_duplicates=0.5, offset=(10, 0, 0))
        assert isinstance(result, Geometry)
        # 配列化により線分が増加
        assert len(result.offsets) >= len(simple_geometry.offsets)
