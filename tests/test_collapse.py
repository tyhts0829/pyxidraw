import pytest
import numpy as np

from effects.collapse import Collapse


class TestCollapse:
    """Collapseエフェクトのテスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される"""
        self.collapse = Collapse()
    
    def test_basic_functionality(self):
        """基本機能のテスト"""
        # 単純な線分を作成
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)
        vertices_list = [vertices]
        
        # エフェクトを適用
        result = self.collapse.apply(vertices_list, intensity=0.1, n_divisions=0.2)
        
        # 結果が空でないことを確認
        assert len(result) > 0
        assert all(isinstance(v, np.ndarray) for v in result)
    
    def test_empty_input(self):
        """空の入力のテスト"""
        result = self.collapse.apply([])
        assert result == []
    
    def test_single_vertex(self):
        """頂点が1つの場合のテスト"""
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        vertices_list = [vertices]
        
        result = self.collapse.apply(vertices_list, intensity=0.1, n_divisions=0.2)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], vertices)
    
    def test_zero_intensity(self):
        """強度が0の場合のテスト"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)
        vertices_list = [vertices]
        
        result = self.collapse.apply(vertices_list, intensity=0.0, n_divisions=0.2)
        # 強度0では変化しないはず
        assert len(result) == len(vertices_list)
    
    def test_zero_divisions(self):
        """分割数が0の場合のテスト"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)
        vertices_list = [vertices]
        
        result = self.collapse.apply(vertices_list, intensity=0.1, n_divisions=0.0)
        # 分割数0では変化しないはず
        assert len(result) == len(vertices_list)
    
    def test_multiple_lines(self):
        """複数の線分のテスト"""
        vertices1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)
        vertices2 = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], dtype=np.float64)
        vertices_list = [vertices1, vertices2]
        
        result = self.collapse.apply(vertices_list, intensity=0.1, n_divisions=0.2)
        assert len(result) >= 2  # 細分化により増える可能性がある
    
    def test_default_parameters(self):
        """デフォルトパラメータのテスト"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)
        vertices_list = [vertices]
        
        # パラメータなしで実行
        result = self.collapse.apply(vertices_list)
        assert len(result) > 0