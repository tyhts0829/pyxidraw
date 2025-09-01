"""
effects.buffer モジュールのテスト
"""
import numpy as np
import pytest

from effects.buffer import Buffer
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


class TestBuffer:
    def test_init(self):
        """Bufferクラスの初期化テスト"""
        effect = Buffer()
        assert effect is not None

    def test_no_buffer(self, simple_geometry):
        """バッファなしテスト（distance=0.0）"""
        effect = Buffer()
        result = effect(simple_geometry, distance=0.0)
        assert isinstance(result, Geometry)

    def test_basic_buffer(self, simple_geometry):
        """基本的なバッファテスト"""
        effect = Buffer()
        result = effect(simple_geometry, distance=0.5)
        assert isinstance(result, Geometry)

    def test_max_buffer(self, simple_geometry):
        """最大バッファテスト（distance=1.0）"""
        effect = Buffer()
        result = effect(simple_geometry, distance=1.0)
        assert isinstance(result, Geometry)

    def test_various_distances(self, simple_geometry):
        """様々な距離テスト"""
        effect = Buffer()
        
        for distance in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = effect(simple_geometry, distance=distance)
            assert isinstance(result, Geometry)

    def test_join_styles(self, square_geometry):
        """接合スタイルテスト"""
        effect = Buffer()
        
        # round style (0.0-0.33)
        result_round = effect(square_geometry, distance=0.5, join_style=0.2)
        assert isinstance(result_round, Geometry)
        
        # mitre style (0.33-0.66)
        result_mitre = effect(square_geometry, distance=0.5, join_style=0.5)
        assert isinstance(result_mitre, Geometry)
        
        # bevel style (0.66-1.0)
        result_bevel = effect(square_geometry, distance=0.5, join_style=0.8)
        assert isinstance(result_bevel, Geometry)

    def test_resolution_levels(self, square_geometry):
        """解像度レベルテスト"""
        effect = Buffer()
        
        for resolution in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = effect(square_geometry, distance=0.5, resolution=resolution)
            assert isinstance(result, Geometry)

    def test_min_resolution(self, simple_geometry):
        """最小解像度テスト"""
        effect = Buffer()
        result = effect(simple_geometry, distance=0.5, resolution=0.0)
        assert isinstance(result, Geometry)

    def test_max_resolution(self, simple_geometry):
        """最大解像度テスト"""
        effect = Buffer()
        result = effect(simple_geometry, distance=0.5, resolution=1.0)
        assert isinstance(result, Geometry)

    def test_combined_parameters(self, square_geometry):
        """パラメータ組み合わせテスト"""
        effect = Buffer()
        result = effect(
            square_geometry,
            distance=0.7,
            join_style=0.3,
            resolution=0.8
        )
        assert isinstance(result, Geometry)

    def test_small_distance(self, simple_geometry):
        """小さな距離テスト"""
        effect = Buffer()
        result = effect(simple_geometry, distance=0.05)
        assert isinstance(result, Geometry)

    def test_edge_join_style_values(self, square_geometry):
        """接合スタイル境界値テスト"""
        effect = Buffer()
        
        # 境界値でのテスト
        for join_style in [0.0, 0.33, 0.66, 1.0]:
            result = effect(square_geometry, distance=0.5, join_style=join_style)
            assert isinstance(result, Geometry)

    def test_single_line_buffer(self):
        """単一線バッファテスト"""
        single_line = [np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)]
        geometry = Geometry.from_lines(single_line)
        
        effect = Buffer()
        result = effect(geometry, distance=0.3)
        assert isinstance(result, Geometry)

    def test_complex_geometry_buffer(self, simple_geometry):
        """複雑なジオメトリバッファテスト"""
        effect = Buffer()
        result = effect(simple_geometry, distance=0.4, join_style=0.5, resolution=0.6)
        assert isinstance(result, Geometry)

    def test_buffer_increases_complexity(self, square_geometry):
        """バッファによる複雑度増加テスト"""
        effect = Buffer()
        result = effect(square_geometry, distance=0.5)
        assert isinstance(result, Geometry)
        # バッファ適用により通常は頂点数が増加
        assert len(result.coords) >= len(square_geometry.coords)

    def test_repeated_buffer(self, simple_geometry):
        """繰り返しバッファテスト"""
        effect = Buffer()
        
        # 最初のバッファ
        result1 = effect(simple_geometry, distance=0.2)
        # 二回目のバッファ
        result2 = effect(result1, distance=0.2)
        
        assert isinstance(result1, Geometry)
        assert isinstance(result2, Geometry)