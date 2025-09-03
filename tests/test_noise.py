"""
effects.noise モジュールのテスト
"""
import numpy as np
import pytest

from effects.noise import noise
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestNoise:
    def test_no_noise(self, simple_geometry):
        """ノイズなしテスト（intensity=0.0）"""
        result = noise(simple_geometry, intensity=0.0)
        assert isinstance(result, Geometry)
        # 強度0なので元と同じ
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_basic_noise(self, simple_geometry):
        """基本的なノイズテスト"""
        result = noise(simple_geometry, intensity=0.1)
        assert isinstance(result, Geometry)
        # ノイズが適用されるので座標が変化
        assert not np.allclose(result.coords, simple_geometry.coords)

    def test_max_noise(self, simple_geometry):
        """最大ノイズテスト（intensity=1.0）"""
        result = noise(simple_geometry, intensity=1.0)
        assert isinstance(result, Geometry)

    def test_various_intensities(self, simple_geometry):
        """様々な強度テスト"""
        for intensity in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = noise(simple_geometry, intensity=intensity)
            assert isinstance(result, Geometry)
            assert len(result.coords) == len(simple_geometry.coords)

    def test_frequency_single_value(self, simple_geometry):
        """単一値周波数テスト"""
        result = noise(simple_geometry, intensity=0.5, frequency=0.3)
        assert isinstance(result, Geometry)

    def test_frequency_tuple(self, simple_geometry):
        """タプル周波数テスト"""
        result = noise(simple_geometry, intensity=0.5, frequency=(0.2, 0.4, 0.6))
        assert isinstance(result, Geometry)

    def test_time_parameter(self, simple_geometry):
        """時間パラメータテスト"""
        result1 = noise(simple_geometry, intensity=0.5, time=0.0)
        result2 = noise(simple_geometry, intensity=0.5, time=1.0)
        
        assert isinstance(result1, Geometry)
        assert isinstance(result2, Geometry)
        # 時間が異なるのでノイズも異なる
        assert not np.allclose(result1.coords, result2.coords)

    def test_reproducibility(self, simple_geometry):
        """再現性テスト"""
        # 同じパラメータで複数回実行
        result1 = noise(simple_geometry, intensity=0.5, frequency=0.3, time=0.5)
        result2 = noise(simple_geometry, intensity=0.5, frequency=0.3, time=0.5)
        
        # 同じパラメータなら同じ結果
        np.testing.assert_allclose(result1.coords, result2.coords, rtol=1e-6)

    def test_high_frequency(self, simple_geometry):
        """高周波数テスト"""
        result = noise(simple_geometry, intensity=0.5, frequency=1.0)
        assert isinstance(result, Geometry)

    def test_low_frequency(self, simple_geometry):
        """低周波数テスト"""
        result = noise(simple_geometry, intensity=0.5, frequency=0.0)
        assert isinstance(result, Geometry)

    def test_mixed_frequency_tuple(self, simple_geometry):
        """混合周波数タプルテスト"""
        result = noise(simple_geometry, intensity=0.5, frequency=(0.0, 0.5, 1.0))
        assert isinstance(result, Geometry)

    def test_negative_time(self, simple_geometry):
        """負の時間テスト"""
        result = noise(simple_geometry, intensity=0.5, time=-1.0)
        assert isinstance(result, Geometry)

    def test_large_time(self, simple_geometry):
        """大きな時間値テスト"""
        result = noise(simple_geometry, intensity=0.5, time=1000.0)
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        result = noise(simple_geometry, intensity=0.5)
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)

    def test_intensity_progression(self, simple_geometry):
        """強度段階テスト"""
        results = []
        for intensity in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            result = noise(simple_geometry, intensity=intensity)
            results.append(result)
        
        # 強度0は元と同じ
        np.testing.assert_allclose(results[0].coords, simple_geometry.coords, rtol=1e-6)
        
        # その他の強度では変化がある
        for i in range(1, len(results)):
            assert not np.allclose(results[i].coords, simple_geometry.coords)
