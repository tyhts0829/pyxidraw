"""
effects.buffer モジュールのテスト
"""
import numpy as np
import pytest

from effects.buffer import offset
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


@pytest.fixture
def square_geometry():
    """テスト用の正方形Geometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    ]
    return Geometry.from_lines(lines)


class TestOffset:
    def test_no_buffer(self, simple_geometry):
        """バッファなしテスト（distance=0.0）"""
        result = offset(simple_geometry, distance=0.0)
        assert isinstance(result, Geometry)

    def test_basic_buffer(self, simple_geometry):
        """基本的なバッファテスト"""
        result = offset(simple_geometry, distance=0.5)
        assert isinstance(result, Geometry)

    def test_max_buffer(self, simple_geometry):
        """最大バッファテスト（distance=1.0）"""
        result = offset(simple_geometry, distance=1.0)
        assert isinstance(result, Geometry)

    def test_various_distances(self, simple_geometry):
        """様々な距離テスト"""
        for distance in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = offset(simple_geometry, distance=distance)
            assert isinstance(result, Geometry)

    def test_join_styles(self, square_geometry):
        """接合スタイルテスト"""
        # round style
        result_round = offset(square_geometry, distance=0.5, join='round')
        assert isinstance(result_round, Geometry)
        
        # mitre style
        result_mitre = offset(square_geometry, distance=0.5, join='mitre')
        assert isinstance(result_mitre, Geometry)
        
        # bevel style
        result_bevel = offset(square_geometry, distance=0.5, join='bevel')
        assert isinstance(result_bevel, Geometry)

    def test_resolution_levels(self, square_geometry):
        """解像度レベルテスト"""
        for resolution in [2, 4, 8, 12, 16]:
            result = offset(square_geometry, distance=0.5, segments_per_circle=resolution)
            assert isinstance(result, Geometry)

    def test_min_resolution(self, simple_geometry):
        """最小解像度テスト"""
        result = offset(simple_geometry, distance=0.5, segments_per_circle=1)
        assert isinstance(result, Geometry)

    def test_max_resolution(self, simple_geometry):
        """最大解像度テスト"""
        result = offset(simple_geometry, distance=0.5, segments_per_circle=64)
        assert isinstance(result, Geometry)

    def test_combined_parameters(self, square_geometry):
        """パラメータ組み合わせテスト"""
        result = offset(
            square_geometry,
            distance=0.7,
            join='round',
            segments_per_circle=12
        )
        assert isinstance(result, Geometry)

    def test_small_distance(self, simple_geometry):
        """小さな距離テスト"""
        result = offset(simple_geometry, distance=0.05)
        assert isinstance(result, Geometry)

    def test_edge_join_style_values(self, square_geometry):
        """接合スタイル境界値テスト"""
        # 境界値でのテスト
        for join in ['mitre', 'round', 'bevel']:
            result = offset(square_geometry, distance=0.5, join=join)
            assert isinstance(result, Geometry)

    def test_single_line_buffer(self):
        """単一線バッファテスト"""
        single_line = [np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)]
        geometry = Geometry.from_lines(single_line)
        
        result = offset(geometry, distance=0.3)
        assert isinstance(result, Geometry)

    def test_complex_geometry_buffer(self, simple_geometry):
        """複雑なジオメトリバッファテスト"""
        result = offset(simple_geometry, distance=0.4, join='round', segments_per_circle=8)
        assert isinstance(result, Geometry)

    def test_buffer_increases_complexity(self, square_geometry):
        """バッファによる複雑度増加テスト"""
        result = offset(square_geometry, distance=0.5)
        assert isinstance(result, Geometry)
        # バッファ適用により通常は頂点数が増加
        assert len(result.coords) >= len(square_geometry.coords)

    def test_repeated_buffer(self, simple_geometry):
        """繰り返しバッファテスト"""
        # 最初のバッファ
        result1 = offset(simple_geometry, distance=0.2)
        # 二回目のバッファ
        result2 = offset(result1, distance=0.2)
        
        assert isinstance(result1, Geometry)
        assert isinstance(result2, Geometry)
