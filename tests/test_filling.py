"""
effects.filling モジュールのテスト
"""
import numpy as np
import pytest

from effects.filling import Filling
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
    """テスト用の正方形Geometry（閉じた形状）"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    ]
    return Geometry.from_lines(lines)


class TestFilling:
    def test_init(self):
        """Fillingクラスの初期化テスト"""
        effect = Filling()
        assert effect is not None

    def test_lines_pattern(self, square_geometry):
        """線パターン塗りつぶしテスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="lines", density=0.5)
        assert isinstance(result, Geometry)

    def test_cross_pattern(self, square_geometry):
        """クロスパターン塗りつぶしテスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="cross", density=0.5)
        assert isinstance(result, Geometry)

    def test_dots_pattern(self, square_geometry):
        """ドットパターン塗りつぶしテスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="dots", density=0.5)
        assert isinstance(result, Geometry)

    def test_min_density(self, square_geometry):
        """最小密度テスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="lines", density=0.0)
        assert isinstance(result, Geometry)

    def test_max_density(self, square_geometry):
        """最大密度テスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="lines", density=1.0)
        assert isinstance(result, Geometry)

    def test_various_densities(self, square_geometry):
        """様々な密度テスト"""
        effect = Filling()
        
        for density in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = effect(square_geometry, pattern="lines", density=density)
            assert isinstance(result, Geometry)

    def test_angle_rotation(self, square_geometry):
        """角度指定テスト"""
        effect = Filling()
        
        # 0度
        result_0 = effect(square_geometry, pattern="lines", density=0.5, angle=0.0)
        assert isinstance(result_0, Geometry)
        
        # 45度
        result_45 = effect(square_geometry, pattern="lines", density=0.5, angle=0.785398)  # π/4
        assert isinstance(result_45, Geometry)
        
        # 90度
        result_90 = effect(square_geometry, pattern="lines", density=0.5, angle=1.570796)  # π/2
        assert isinstance(result_90, Geometry)

    def test_all_patterns_with_angles(self, square_geometry):
        """全パターンと角度の組み合わせテスト"""
        effect = Filling()
        patterns = ["lines", "cross", "dots"]
        
        for pattern in patterns:
            result = effect(square_geometry, pattern=pattern, density=0.5, angle=0.5)
            assert isinstance(result, Geometry)

    def test_simple_geometry_filling(self, simple_geometry):
        """単純なジオメトリの塗りつぶしテスト"""
        effect = Filling()
        result = effect(simple_geometry, pattern="lines", density=0.5)
        assert isinstance(result, Geometry)

    def test_invalid_pattern_fallback(self, square_geometry):
        """無効なパターンのフォールバックテスト"""
        effect = Filling()
        # 無効なパターンを指定しても例外が起きないことを確認
        result = effect(square_geometry, pattern="invalid", density=0.5)
        assert isinstance(result, Geometry)

    def test_high_density_performance(self, square_geometry):
        """高密度でのパフォーマンステスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="cross", density=0.95)
        assert isinstance(result, Geometry)

    def test_low_density_performance(self, square_geometry):
        """低密度でのパフォーマンステスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="dots", density=0.05)
        assert isinstance(result, Geometry)

    def test_negative_angle(self, square_geometry):
        """負の角度テスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="lines", density=0.5, angle=-0.785398)
        assert isinstance(result, Geometry)

    def test_large_angle(self, square_geometry):
        """大きな角度テスト"""
        effect = Filling()
        result = effect(square_geometry, pattern="lines", density=0.5, angle=6.283185)  # 2π
        assert isinstance(result, Geometry)