"""
effects.repeat モジュールのテスト
"""
import numpy as np
import pytest

from effects.repeat import repeat
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestRepeat:
    def test_no_duplicates(self, simple_geometry):
        """複製なしテスト（n_duplicates=0.0）"""
        result = repeat(simple_geometry, count=0)
        assert isinstance(result, Geometry)
        # 複製なしなので元と同じかそれに近い
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_basic_array(self, simple_geometry):
        """基本的な配列テスト"""
        result = repeat(simple_geometry, count=5, offset=(10, 0, 0))
        assert isinstance(result, Geometry)
        # 複製により頂点数が増加
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_max_duplicates(self, simple_geometry):
        """最大複製テスト（n_duplicates=1.0）"""
        result = repeat(simple_geometry, count=10, offset=(5, 5, 0))
        assert isinstance(result, Geometry)

    def test_x_offset_only(self, simple_geometry):
        """X軸オフセットのみテスト"""
        result = repeat(simple_geometry, count=3, offset=(10, 0, 0))
        assert isinstance(result, Geometry)

    def test_y_offset_only(self, simple_geometry):
        """Y軸オフセットのみテスト"""
        result = repeat(simple_geometry, count=3, offset=(0, 10, 0))
        assert isinstance(result, Geometry)

    def test_z_offset_only(self, simple_geometry):
        """Z軸オフセットのみテスト"""
        result = repeat(simple_geometry, count=3, offset=(0, 0, 10))
        assert isinstance(result, Geometry)

    def test_diagonal_offset(self, simple_geometry):
        """対角線オフセットテスト"""
        result = repeat(simple_geometry, count=4, offset=(5, 5, 5))
        assert isinstance(result, Geometry)

    def test_with_rotation(self, simple_geometry):
        """回転付き配列テスト"""
        result = repeat(
            simple_geometry,
            count=5,
            offset=(10, 0, 0),
            angles_rad_step=(0.2 * np.pi, 0.0, 0.0)
        )
        assert isinstance(result, Geometry)

    def test_with_scaling(self, simple_geometry):
        """スケーリング付き配列テスト"""
        result = repeat(
            simple_geometry,
            count=5,
            offset=(10, 0, 0),
            scale=(0.3, 0.3, 0.3)
        )
        assert isinstance(result, Geometry)

    def test_with_center(self, simple_geometry):
        """中心点指定配列テスト"""
        result = repeat(
            simple_geometry,
            count=5,
            offset=(10, 0, 0),
            pivot=(5, 5, 0)
        )
        assert isinstance(result, Geometry)

    def test_combined_transformations(self, simple_geometry):
        """複合変換配列テスト"""
        result = repeat(
            simple_geometry,
            count=6,
            offset=(8, 3, 2),
            angles_rad_step=(0.4 * np.pi, 0.2 * np.pi, 0.6 * np.pi),
            scale=(0.4, 0.6, 0.8),
            pivot=(1, 1, 1)
        )
        assert isinstance(result, Geometry)

    def test_negative_offset(self, simple_geometry):
        """負のオフセットテスト"""
        result = repeat(simple_geometry, count=3, offset=(-5, -3, -2))
        assert isinstance(result, Geometry)

    def test_zero_offset(self, simple_geometry):
        """ゼロオフセットテスト"""
        result = repeat(simple_geometry, count=3, offset=(0, 0, 0))
        assert isinstance(result, Geometry)

    def test_large_offset(self, simple_geometry):
        """大きなオフセットテスト"""
        result = repeat(simple_geometry, count=3, offset=(100, 200, 50))
        assert isinstance(result, Geometry)

    def test_various_duplicate_counts(self, simple_geometry):
        """様々な複製数テスト"""
        for n_duplicates in [0.1, 0.3, 0.5, 0.7, 0.9]:
            count = int(round(n_duplicates * 10))
            result = repeat(simple_geometry, count=count, offset=(5, 0, 0))
            assert isinstance(result, Geometry)

    def test_rotation_variations(self, simple_geometry):
        """回転バリエーションテスト"""
        # X軸回転
        result_x = repeat(simple_geometry, count=3, offset=(5, 0, 0), angles_rad_step=(0.4 * np.pi, 0.0, 0.0))
        # Y軸回転
        result_y = repeat(simple_geometry, count=3, offset=(5, 0, 0), angles_rad_step=(0.0, 0.4 * np.pi, 0.0))
        # Z軸回転
        result_z = repeat(simple_geometry, count=3, offset=(5, 0, 0), angles_rad_step=(0.0, 0.0, 0.4 * np.pi))
        
        assert isinstance(result_x, Geometry)
        assert isinstance(result_y, Geometry)
        assert isinstance(result_z, Geometry)

    def test_scale_variations(self, simple_geometry):
        """スケールバリエーションテスト"""
        # 拡大
        result_expand = repeat(simple_geometry, count=3, offset=(5, 0, 0), scale=(0.8, 0.8, 0.8))
        # 縮小
        result_shrink = repeat(simple_geometry, count=3, offset=(5, 0, 0), scale=(0.2, 0.2, 0.2))
        
        assert isinstance(result_expand, Geometry)
        assert isinstance(result_shrink, Geometry)

    def test_preserves_original_structure(self, simple_geometry):
        """元構造保持テスト"""
        result = repeat(simple_geometry, count=5, offset=(10, 0, 0))
        assert isinstance(result, Geometry)
        # 配列化により線分が増加
        assert len(result.offsets) >= len(simple_geometry.offsets)
