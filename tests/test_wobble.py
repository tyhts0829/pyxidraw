import numpy as np
import pytest

from api.shapes import polygon
from effects.wobble import Wobble


class TestWobble:
    """Wobble エフェクトのテスト"""
    
    def test_basic_wobble(self):
        """基本的なウォブルエフェクトのテスト"""
        # 正方形を作成
        vertices_list = polygon(4)
        wobble = Wobble()
        
        # デフォルトパラメータでウォブルを適用
        result = wobble.apply(vertices_list)
        
        # 頂点数は変わらない
        assert len(result) == len(vertices_list)
        for original, wobbled in zip(vertices_list, result):
            assert len(original) == len(wobbled)
            assert original.shape == wobbled.shape
            
            # ウォブルにより座標が変化している
            assert not np.allclose(original, wobbled)
    
    def test_wobble_with_amplitude_zero(self):
        """振幅が0の場合は変化しない"""
        vertices_list = polygon(6)
        wobble = Wobble()
        
        result = wobble.apply(vertices_list, amplitude=0.0)
        
        # 振幅が0なので元の形状と同じ
        for original, wobbled in zip(vertices_list, result):
            np.testing.assert_array_almost_equal(original, wobbled)
    
    def test_wobble_with_single_frequency(self):
        """単一の周波数値を指定"""
        vertices_list = polygon(4)
        wobble = Wobble()
        
        # float で周波数を指定
        result = wobble.apply(vertices_list, frequency=0.5, amplitude=0.1)
        
        assert len(result) == len(vertices_list)
        for original, wobbled in zip(vertices_list, result):
            assert not np.allclose(original, wobbled)
    
    def test_wobble_with_tuple_frequency(self):
        """タプルで各軸の周波数を個別指定"""
        vertices_list = polygon(4)
        wobble = Wobble()
        
        # 各軸に異なる周波数を指定
        result = wobble.apply(vertices_list, frequency=(0.1, 0.2, 0.3), amplitude=0.1)
        
        assert len(result) == len(vertices_list)
        for original, wobbled in zip(vertices_list, result):
            assert not np.allclose(original, wobbled)
    
    def test_wobble_with_phase(self):
        """位相オフセットのテスト"""
        vertices_list = polygon(8)
        wobble = Wobble()
        
        # 位相0の結果
        result1 = wobble.apply(vertices_list, phase=0.0, amplitude=0.1)
        # 位相πの結果
        result2 = wobble.apply(vertices_list, phase=np.pi, amplitude=0.1)
        
        # 異なる位相では異なる結果になる
        for v1, v2 in zip(result1, result2):
            assert not np.allclose(v1, v2)
    
    def test_empty_list(self):
        """空のリストの処理"""
        wobble = Wobble()
        result = wobble.apply([])
        assert result == []
    
    def test_empty_vertices_in_list(self):
        """空の頂点配列を含むリストの処理"""
        vertices_list = [np.array([]), np.array([[1, 0], [0, 1]]), np.array([])]
        wobble = Wobble()
        
        result = wobble.apply(vertices_list)
        
        assert len(result) == 3
        assert len(result[0]) == 0
        assert len(result[2]) == 0
        assert not np.allclose(vertices_list[1], result[1])
    
    def test_multiple_shapes(self):
        """複数の形状に対するウォブル"""
        vertices_list = polygon(4) + polygon(6)
        wobble = Wobble()
        
        result = wobble.apply(vertices_list, amplitude=0.2, frequency=0.3)
        
        assert len(result) == len(vertices_list)
        for original, wobbled in zip(vertices_list, result):
            if len(original) > 0:
                assert not np.allclose(original, wobbled)
    
    def test_3d_vertices(self):
        """3D頂点に対するウォブル"""
        # 3D頂点を作成（z座標を非ゼロにする）
        vertices = np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]], dtype=np.float32)
        vertices_list = [vertices]
        wobble = Wobble()
        
        result = wobble.apply(vertices_list, amplitude=0.1, frequency=(0.1, 0.2, 0.3))
        
        assert len(result) == 1
        assert result[0].shape == vertices.shape
        # z座標も変化している（元がゼロでない場合）
        assert not np.allclose(vertices[:, 2], result[0][:, 2])
    
    def test_invalid_frequency_tuple_length(self):
        """不正な長さの周波数タプル"""
        vertices_list = polygon(4)
        wobble = Wobble()
        
        # 長さが3でないタプルを指定するとデフォルト値になる
        result = wobble.apply(vertices_list, frequency=(0.1, 0.2), amplitude=0.1)
        
        # デフォルト値で処理される
        assert len(result) == len(vertices_list)
    
    def test_large_amplitude(self):
        """大きな振幅での動作確認"""
        vertices_list = polygon(3)
        wobble = Wobble()
        
        result = wobble.apply(vertices_list, amplitude=10.0, frequency=0.1)
        
        # 大きな変位が加わる
        for original, wobbled in zip(vertices_list, result):
            if len(original) > 0:
                # 少なくとも1つの頂点は大きく移動する
                max_displacement = np.max(np.abs(wobbled - original))
                assert max_displacement > 1.0
    
    def test_high_frequency(self):
        """高周波数での動作確認"""
        # 座標値がsin関数で明確な変化を示すように調整
        vertices = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], dtype=np.float32)
        vertices_list = [vertices]
        wobble = Wobble()
        
        # sin(2π * 1.0 * 0.25) = sin(π/2) = 1.0 で最大変位を得る
        result = wobble.apply(vertices_list, amplitude=0.5, frequency=1.0)
        
        assert len(result) == len(vertices_list)
        for original, wobbled in zip(vertices_list, result):
            # 十分大きな変化があることを確認
            max_diff = np.max(np.abs(wobbled - original))
            assert max_diff > 0.1
    
    def test_negative_amplitude(self):
        """負の振幅でも動作する"""
        vertices_list = polygon(5)
        wobble = Wobble()
        
        result = wobble.apply(vertices_list, amplitude=-0.1, frequency=0.5)
        
        assert len(result) == len(vertices_list)
        for original, wobbled in zip(vertices_list, result):
            if len(original) > 0:
                assert not np.allclose(original, wobbled)