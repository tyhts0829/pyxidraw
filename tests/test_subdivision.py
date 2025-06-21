import pytest
import numpy as np
from effects.subdivision import Subdivision, _subdivide_core


class TestSubdivision:
    """Subdivisionクラスのテスト"""
    
    def setup_method(self):
        """テストの前処理"""
        self.subdivision = Subdivision()
    
    def test_apply_basic(self):
        """基本的な細分化テスト"""
        # 2点の線分
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        vertices_list = [vertices]
        
        result = self.subdivision.apply(vertices_list, n_divisions=0.1)  # 1回分割
        
        # 結果の検証
        assert len(result) == 1
        assert len(result[0]) == 3  # 2点 -> 3点
        
        # 元の点が保持されているか
        np.testing.assert_array_almost_equal(result[0][0], [0, 0, 0])
        np.testing.assert_array_almost_equal(result[0][2], [1, 0, 0])
        
        # 中点が正しく追加されているか
        np.testing.assert_array_almost_equal(result[0][1], [0.5, 0, 0])
    
    def test_apply_multiple_divisions(self):
        """複数回の細分化テスト"""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        vertices_list = [vertices]
        
        result = self.subdivision.apply(vertices_list, n_divisions=2)
        
        # 2回の細分化: 2点 -> 3点 -> 5点
        assert len(result[0]) == 5
        
        # 中点が正しく配置されているか
        expected = np.array([
            [0.0, 0, 0],    # 元の点
            [0.25, 0, 0],   # 中点
            [0.5, 0, 0],    # 中点
            [0.75, 0, 0],   # 中点
            [1.0, 0, 0]     # 元の点
        ])
        np.testing.assert_array_almost_equal(result[0], expected)
    
    def test_apply_empty_list(self):
        """空のリストの処理テスト"""
        result = self.subdivision.apply([], n_divisions=0.5)
        assert result == []
    
    def test_apply_single_point(self):
        """単一点の処理テスト"""
        vertices = np.array([[0, 0, 0]], dtype=np.float32)
        vertices_list = [vertices]
        
        result = self.subdivision.apply(vertices_list, n_divisions=0.5)
        
        # 単一点は変更されない
        assert len(result[0]) == 1
        np.testing.assert_array_almost_equal(result[0], vertices)
    
    def test_apply_multiple_lines(self):
        """複数の線の処理テスト"""
        line1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        line2 = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
        vertices_list = [line1, line2]
        
        result = self.subdivision.apply(vertices_list, n_divisions=0.1)  # 1回分割
        
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        
        # 各線の中点が正しく追加されているか
        np.testing.assert_array_almost_equal(result[0][1], [0.5, 0, 0])
        np.testing.assert_array_almost_equal(result[1][1], [0.5, 1, 0])
    
    def test_apply_3d_coordinates(self):
        """3D座標の処理テスト"""
        vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        vertices_list = [vertices]
        
        result = self.subdivision.apply(vertices_list, n_divisions=0.1)  # 1回分割
        
        assert len(result[0]) == 3
        np.testing.assert_array_almost_equal(result[0][1], [0.5, 0.5, 0.5])
    
    def test_apply_max_divisions_limit(self):
        """最大分割数制限のテスト"""
        # 大きな線分を使用して最小長チェックを回避
        vertices = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
        vertices_list = [vertices]
        
        # 制限を超える分割数でテスト
        result = self.subdivision.apply(vertices_list, n_divisions=1.0)  # 最大分割
        
        # 最大10回の分割制限が適用されるか
        # 実際は9回または10回の分割で最小長チェックで停止
        # 結果として1025点程度になる
        assert len(result[0]) >= 1000  # 十分多くの点が生成されることを確認
        assert len(result[0]) <= 2048  # 制限内に収まることを確認
    
    def test_apply_minimum_length_check(self):
        """最小長チェックのテスト"""
        # 非常に近い2点
        vertices = np.array([[0, 0, 0], [0.005, 0, 0]], dtype=np.float32)
        vertices_list = [vertices]
        
        result = self.subdivision.apply(vertices_list, n_divisions=0.5)
        
        # 最小長以下の場合は分割されない
        assert len(result[0]) == 2
        np.testing.assert_array_almost_equal(result[0], vertices)
    
    def test_apply_zero_divisions(self):
        """0.0の場合は何もしないテスト"""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        vertices_list = [vertices]
        
        result = self.subdivision.apply(vertices_list, n_divisions=0.0)
        
        # 0.0の場合は元の配列がそのまま返される
        assert len(result) == 1
        assert len(result[0]) == 2
        np.testing.assert_array_almost_equal(result[0], vertices)


class TestSubdivisionNumbaFunctions:
    """Numba最適化関数のテスト"""
    
    def test_subdivide_core_basic(self):
        """_subdivide_core関数の基本テスト"""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        
        result = _subdivide_core(vertices, 1)
        
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result[1], [0.5, 0, 0])
    
    def test_subdivide_core_multiple_divisions(self):
        """_subdivide_core関数の複数分割テスト"""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        
        result = _subdivide_core(vertices, 2)
        
        assert len(result) == 5
        expected = np.array([
            [0.0, 0, 0],
            [0.25, 0, 0],
            [0.5, 0, 0],
            [0.75, 0, 0],
            [1.0, 0, 0]
        ])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_subdivide_core_single_point(self):
        """_subdivide_core関数の単一点テスト"""
        vertices = np.array([[0, 0, 0]], dtype=np.float64)
        
        result = _subdivide_core(vertices, 1)
        
        # 単一点は変更されない
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result, vertices)
    
    def test_subdivide_core_min_length(self):
        """_subdivide_core関数の最小長テスト"""
        # 非常に近い2点
        vertices = np.array([[0, 0, 0], [0.005, 0, 0]], dtype=np.float64)
        
        result = _subdivide_core(vertices, 1)
        
        # 最小長以下の場合は分割されない
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result, vertices)
    


class TestSubdivisionPerformance:
    """パフォーマンステスト"""
    
    def test_large_dataset_performance(self):
        """大きなデータセットでのパフォーマンステスト"""
        import time
        
        # 大きなデータセットを作成
        vertices_list = []
        for i in range(100):
            vertices = np.random.rand(10, 3).astype(np.float64)
            vertices_list.append(vertices)
        
        subdivision = Subdivision()
        
        # 実行時間測定
        start_time = time.time()
        result = subdivision.apply(vertices_list, n_divisions=0.2)  # 2回分割相当
        end_time = time.time()
        
        # 結果の検証
        assert len(result) == 100
        assert all(len(line) > 10 for line in result)  # 各線が分割されているか
        
        # パフォーマンス確認（5秒以内に完了）
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"実行時間が長すぎます: {execution_time:.2f}秒"
        
        print(f"大きなデータセットの処理時間: {execution_time:.3f}秒")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])