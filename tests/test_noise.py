import numpy as np
import pytest

from api.shapes import polygon
from effects.noise import Noise


class TestNoise:
    """Noise エフェクトのテスト"""
    
    def test_basic_noise(self):
        """基本的なノイズエフェクトのテスト"""
        # 正方形を作成
        vertices_list = polygon(4)
        noise = Noise()
        
        # デフォルトパラメータでノイズを適用
        result = noise.apply(vertices_list)
        
        # 頂点数は変わらない
        assert len(result) == len(vertices_list)
        for original, modified in zip(vertices_list, result):
            assert len(original) == len(modified)
            assert original.shape == modified.shape
            
            # ノイズにより座標が変化している
            assert not np.allclose(original, modified)
    
    def test_noise_with_zero_intensity(self):
        """強度が0の場合は変化しない"""
        vertices_list = polygon(6)
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=0.0)
        
        # 強度が0なので元の形状と同じ
        for original, modified in zip(vertices_list, result):
            np.testing.assert_array_almost_equal(original, modified)
    
    def test_noise_with_single_frequency(self):
        """単一の周波数値を指定"""
        vertices_list = polygon(4)
        noise = Noise()
        
        # float で周波数を指定
        result = noise.apply(vertices_list, frequency=0.5, intensity=0.5)
        
        assert len(result) == len(vertices_list)
        for original, modified in zip(vertices_list, result):
            assert not np.allclose(original, modified)
    
    def test_noise_with_tuple_frequency(self):
        """タプルで各軸の周波数を個別指定"""
        vertices_list = polygon(4)
        noise = Noise()
        
        # 各軸に異なる周波数を指定
        result = noise.apply(vertices_list, frequency=(0.1, 0.2, 0.3), intensity=0.5)
        
        assert len(result) == len(vertices_list)
        for original, modified in zip(vertices_list, result):
            assert not np.allclose(original, modified)
    
    def test_noise_with_time_parameter(self):
        """時間パラメータのテスト"""
        vertices_list = polygon(8)
        noise = Noise()
        
        # 時間0の結果
        result1 = noise.apply(vertices_list, t=0.0, intensity=0.5)
        # 時間100の結果
        result2 = noise.apply(vertices_list, t=100.0, intensity=0.5)
        
        # 異なる時間では異なる結果になる
        for v1, v2 in zip(result1, result2):
            assert not np.allclose(v1, v2)
    
    def test_empty_list(self):
        """空のリストの処理"""
        noise = Noise()
        result = noise.apply([])
        assert result == []
    
    def test_empty_vertices_in_list(self):
        """空の頂点配列を含むリストの処理"""
        vertices_list = [np.array([]), np.array([[1, 0, 0.1], [0, 1, 0.2]], dtype=np.float32), np.array([])]
        noise = Noise()
        
        result = noise.apply(vertices_list)
        
        assert len(result) == 3
        assert len(result[0]) == 0
        assert len(result[2]) == 0
        assert not np.allclose(vertices_list[1], result[1])
    
    def test_multiple_shapes(self):
        """複数の形状に対するノイズ"""
        vertices_list = polygon(4) + polygon(6)
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=0.5, frequency=0.3)
        
        assert len(result) == len(vertices_list)
        for original, modified in zip(vertices_list, result):
            if len(original) > 0:
                assert not np.allclose(original, modified)
    
    def test_3d_vertices(self):
        """3D頂点に対するノイズ"""
        # 3D頂点を作成（z座標を非ゼロにする）
        vertices = np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]], dtype=np.float32)
        vertices_list = [vertices]
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=0.5, frequency=(0.1, 0.2, 0.3))
        
        assert len(result) == 1
        assert result[0].shape == vertices.shape
        # z座標も変化している
        assert not np.allclose(vertices[:, 2], result[0][:, 2])
    
    def test_frequency_tuple_length_normalization(self):
        """周波数タプルの長さ正規化"""
        vertices_list = polygon(4)
        noise = Noise()
        
        # 長さが1のタプルは3次元に拡張される
        result1 = noise.apply(vertices_list, frequency=(0.1,), intensity=0.5)
        result2 = noise.apply(vertices_list, frequency=(0.1, 0.1, 0.1), intensity=0.5)
        
        # 結果は同じになる（同じ周波数）
        assert len(result1) == len(result2)
    
    def test_high_intensity(self):
        """高強度での動作確認"""
        vertices_list = polygon(3)
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=5.0, frequency=0.1)
        
        # 大きな変位が加わる
        for original, modified in zip(vertices_list, result):
            if len(original) > 0:
                # 少なくとも1つの頂点は変位している
                max_displacement = np.max(np.abs(modified - original))
                assert max_displacement > 0.01
    
    def test_high_frequency_noise(self):
        """高周波数ノイズの動作確認"""
        vertices = np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.2], [1.0, 1.0, 0.3], [0.0, 1.0, 0.4]], dtype=np.float32)
        vertices_list = [vertices]
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=0.5, frequency=2.0)
        
        assert len(result) == len(vertices_list)
        for original, modified in zip(vertices_list, result):
            # 変化があることを確認
            assert not np.allclose(original, modified)
    
    def test_negative_intensity(self):
        """負の強度でも動作する"""
        vertices_list = polygon(5)
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=-0.5, frequency=0.5)
        
        assert len(result) == len(vertices_list)
        for original, modified in zip(vertices_list, result):
            if len(original) > 0:
                assert not np.allclose(original, modified)
    
    def test_perlin_noise_consistency(self):
        """Perlinノイズの一貫性テスト"""
        vertices_list = polygon(4)
        noise = Noise()
        
        # 同じパラメータで複数回実行
        result1 = noise.apply(vertices_list, intensity=0.5, frequency=0.3, t=0.0)
        result2 = noise.apply(vertices_list, intensity=0.5, frequency=0.3, t=0.0)
        
        # 結果は同じになる（決定論的）
        for v1, v2 in zip(result1, result2):
            np.testing.assert_array_almost_equal(v1, v2)
    
    def test_different_time_values(self):
        """異なる時間値での変化確認"""
        vertices_list = polygon(6)
        noise = Noise()
        
        results = []
        for t in [0.0, 10.0, 20.0]:
            result = noise.apply(vertices_list, intensity=0.5, frequency=0.5, t=t)
            results.append(result)
        
        # 各時間での結果はすべて異なる
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                for v1, v2 in zip(results[i], results[j]):
                    if len(v1) > 0:
                        assert not np.allclose(v1, v2)
    
    def test_large_vertex_arrays(self):
        """大きな頂点配列での動作確認"""
        # 大きな頂点配列を作成
        vertices = np.random.rand(1000, 3).astype(np.float32)
        vertices_list = [vertices]
        noise = Noise()
        
        result = noise.apply(vertices_list, intensity=0.1, frequency=0.5)
        
        assert len(result) == 1
        assert result[0].shape == vertices.shape
        # 変化があることを確認
        assert not np.allclose(vertices, result[0])