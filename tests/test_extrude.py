"""
effects.extrude モジュールのテスト
"""
import numpy as np
import pytest

from effects.extrude import Extrude
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


class TestExtrude:
    def test_init(self):
        """Extrudeクラスの初期化テスト"""
        effect = Extrude()
        assert effect is not None

    def test_no_extrusion(self, simple_geometry):
        """押し出しなしテスト（distance=0.0）"""
        effect = Extrude()
        result = effect(simple_geometry, distance=0.0)
        assert isinstance(result, Geometry)

    def test_basic_extrusion(self, square_geometry):
        """基本的な押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, direction=(0, 0, 1))
        assert isinstance(result, Geometry)
        # 押し出しにより線分が増加
        assert len(result.coords) > len(square_geometry.coords)

    def test_z_direction_extrusion(self, square_geometry):
        """Z方向押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.8, direction=(0, 0, 1))
        assert isinstance(result, Geometry)

    def test_x_direction_extrusion(self, square_geometry):
        """X方向押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.6, direction=(1, 0, 0))
        assert isinstance(result, Geometry)

    def test_diagonal_direction_extrusion(self, square_geometry):
        """対角線方向押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, direction=(1, 1, 1))
        assert isinstance(result, Geometry)

    def test_scaling_during_extrusion(self, square_geometry):
        """押し出し時のスケーリングテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, scale=0.3)
        assert isinstance(result, Geometry)

    def test_subdivisions_during_extrusion(self, square_geometry):
        """押し出し時の細分化テスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, subdivisions=0.8)
        assert isinstance(result, Geometry)

    def test_max_distance(self, square_geometry):
        """最大距離押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=1.0)
        assert isinstance(result, Geometry)

    def test_min_scale(self, square_geometry):
        """最小スケール押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, scale=0.0)
        assert isinstance(result, Geometry)

    def test_max_scale(self, square_geometry):
        """最大スケール押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, scale=1.0)
        assert isinstance(result, Geometry)

    def test_max_subdivisions(self, square_geometry):
        """最大細分化押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, subdivisions=1.0)
        assert isinstance(result, Geometry)

    def test_combined_parameters(self, square_geometry):
        """パラメータ組み合わせテスト"""
        effect = Extrude()
        result = effect(
            square_geometry,
            distance=0.7,
            direction=(0.5, 0.5, 1.0),
            scale=0.4,
            subdivisions=0.6
        )
        assert isinstance(result, Geometry)

    def test_negative_direction(self, square_geometry):
        """負の方向押し出しテスト"""
        effect = Extrude()
        result = effect(square_geometry, distance=0.5, direction=(0, 0, -1))
        assert isinstance(result, Geometry)

    def test_simple_line_extrusion(self, simple_geometry):
        """単純な線の押し出しテスト"""
        effect = Extrude()
        result = effect(simple_geometry, distance=0.5)
        assert isinstance(result, Geometry)