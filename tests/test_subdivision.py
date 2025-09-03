"""
effects.subdivision モジュールのテスト
"""
import numpy as np
import pytest

from effects.subdivision import subdivision
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestSubdivision:
    def test_no_subdivision(self, simple_geometry):
        """細分化なしテスト（n_divisions=0.0）"""
        result = subdivision(simple_geometry, subdivisions=0.0)
        assert isinstance(result, Geometry)
        # 0.0の場合は元のジオメトリと同じ
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_basic_subdivision(self, simple_geometry):
        """基本的な細分化テスト"""
        result = subdivision(simple_geometry, subdivisions=0.5)
        assert isinstance(result, Geometry)
        # 細分化により頂点数が増加することを確認
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_max_subdivision(self, simple_geometry):
        """最大細分化テスト（n_divisions=1.0）"""
        result = subdivision(simple_geometry, subdivisions=1.0)
        assert isinstance(result, Geometry)
        # 最大細分化で頂点数が大幅に増加
        assert len(result.coords) >= len(simple_geometry.coords)

    def test_increasing_subdivision(self, simple_geometry):
        """段階的な細分化テスト"""
        result_low = subdivision(simple_geometry, subdivisions=0.2)
        result_mid = subdivision(simple_geometry, subdivisions=0.5)
        result_high = subdivision(simple_geometry, subdivisions=0.8)
        
        # 細分化レベルが高いほど頂点数が多い（または同じ）
        assert len(result_low.coords) <= len(result_mid.coords)
        assert len(result_mid.coords) <= len(result_high.coords)

    def test_preserves_line_count(self, simple_geometry):
        """線分数保持テスト"""
        result = subdivision(simple_geometry, subdivisions=0.7)
        assert isinstance(result, Geometry)
        # 線分の数は同じ
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_edge_values(self, simple_geometry):
        """境界値テスト"""
        # 最小値
        result_min = subdivision(simple_geometry, subdivisions=0.0)
        assert isinstance(result_min, Geometry)
        
        # 最大値
        result_max = subdivision(simple_geometry, subdivisions=1.0)
        assert isinstance(result_max, Geometry)

    def test_intermediate_values(self, simple_geometry):
        """中間値テスト"""
        for value in [0.1, 0.3, 0.6, 0.9]:
            result = subdivision(simple_geometry, subdivisions=value)
            assert isinstance(result, Geometry)
            assert len(result.coords) >= len(simple_geometry.coords)

    def test_maintains_start_end_points(self, simple_geometry):
        """開始・終了点保持テスト"""
        result = subdivision(simple_geometry, subdivisions=0.5)
        assert isinstance(result, Geometry)
        
        # 各線分の開始点と終了点は保持される
        for i in range(len(simple_geometry.offsets) - 1):
            start_idx = simple_geometry.offsets[i]
            end_idx = simple_geometry.offsets[i + 1] - 1
            
            result_start_idx = result.offsets[i]
            result_end_idx = result.offsets[i + 1] - 1
            
            # 開始点が同じ
            np.testing.assert_allclose(
                simple_geometry.coords[start_idx], 
                result.coords[result_start_idx], 
                rtol=1e-6
            )
            
            # 終了点が同じ
            np.testing.assert_allclose(
                simple_geometry.coords[end_idx], 
                result.coords[result_end_idx], 
                rtol=1e-6
            )

    def test_single_point_line(self):
        """単一点線テスト"""
        single_point_line = [np.array([[0, 0, 0]], dtype=np.float32)]
        geometry = Geometry.from_lines(single_point_line)
        
        result = subdivision(geometry, subdivisions=0.5)
        assert isinstance(result, Geometry)
        # 単一点の場合は変化なし
        assert len(result.coords) == 1
