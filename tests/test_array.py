import numpy as np
import pytest

from effects.array import Array
from api.shapes import polygon, grid


class TestArray:
    """Arrayエフェクトのテストクラス"""

    def test_basic_array(self):
        """基本的な配列生成のテスト"""
        array = Array()
        square = polygon(4, scale=(50, 50, 0))
        
        # 3個の複製を作成
        result = array.apply(square, n_duplicates=0.3, offset=(10, 10, 0))
        
        # 元の1個 + 3個の複製 = 4個分の頂点があることを確認
        total_vertices = sum(len(vertices) for vertices in result)
        expected_vertices = len(square[0]) * 3  # 複製のみ
        assert total_vertices == expected_vertices

    def test_array_with_rotation(self):
        """回転を伴う配列生成のテスト"""
        array = Array()
        triangle = polygon(3, scale=(30, 30, 0))
        
        # 回転を加えながら2個の複製を作成
        result = array.apply(
            triangle, 
            n_duplicates=0.2, 
            offset=(50, 0, 0),
            rotate=(0.6, 0.5, 0.5)  # 若干の回転
        )
        
        assert len(result) == 2  # 2個の複製
        
        # 各複製が異なる位置・向きにあることを確認
        assert not np.allclose(result[0], result[1])

    def test_array_with_scaling(self):
        """スケーリングを伴う配列生成のテスト"""
        array = Array()
        square = polygon(4, scale=(40, 40, 0))
        
        # スケールを小さくしながら複製
        result = array.apply(
            square,
            n_duplicates=0.3,
            offset=(20, 20, 0),
            scale=(0.8, 0.8, 0.8)  # 80%ずつ縮小
        )
        
        assert len(result) == 3  # 3個の複製
        
        # 各複製のサイズが異なることを確認（重心からの距離で検証）
        sizes = []
        for vertices in result:
            center = np.mean(vertices, axis=0)
            avg_dist = np.mean(np.linalg.norm(vertices - center, axis=1))
            sizes.append(avg_dist)
        
        # サイズが順に小さくなっていることを確認
        assert sizes[0] > sizes[1] > sizes[2]

    def test_array_with_center(self):
        """中心点を指定した配列生成のテスト"""
        array = Array()
        square = polygon(4, scale=(30, 30, 0))
        
        # 中心点を指定して配列生成
        center = (100, 100, 0)
        result = array.apply(
            square,
            n_duplicates=0.2,
            offset=(50, 0, 0),
            center=center
        )
        
        assert len(result) == 2

    def test_zero_duplicates(self):
        """複製数が0の場合のテスト"""
        array = Array()
        square = polygon(4, scale=(50, 50, 0))
        
        # n_duplicates=0の場合、元のリストがそのまま返される
        result = array.apply(square, n_duplicates=0)
        
        assert result == square

    def test_max_duplicates(self):
        """最大複製数のテスト"""
        array = Array()
        square = polygon(4, scale=(20, 20, 0))
        
        # n_duplicates=1.0で最大10個の複製
        result = array.apply(
            square,
            n_duplicates=1.0,
            offset=(15, 0, 0)
        )
        
        assert len(result) == Array.MAX_DUPLICATES

    def test_empty_list(self):
        """空のリストの処理テスト"""
        array = Array()
        result = array.apply([], n_duplicates=0.5)
        assert result == []

    def test_multiple_shapes(self):
        """複数の形状を含むリストの配列生成テスト"""
        array = Array()
        square = polygon(4, scale=(30, 30, 0))
        triangle = polygon(3, scale=(20, 20, 0))
        
        # 複数の形状を一度に配列化
        result = array.apply(
            [square[0], triangle[0]],
            n_duplicates=0.2,
            offset=(40, 0, 0)
        )
        
        # 各形状が2個ずつ複製されることを確認
        assert len(result) == 4  # (square + triangle) * 2個

    def test_3d_array(self):
        """3次元空間での配列生成テスト"""
        array = Array()
        square = polygon(4, scale=(30, 30, 0))
        
        # Z方向にもオフセットを持つ配列
        result = array.apply(
            square,
            n_duplicates=0.3,
            offset=(20, 20, 20),
            rotate=(0.5, 0.6, 0.5),
            scale=(0.9, 0.9, 0.9)
        )
        
        assert len(result) == 3
        
        # Z座標が変化していることを確認
        z_coords = [np.mean(vertices[:, 2]) for vertices in result]
        assert not all(z == z_coords[0] for z in z_coords)

    def test_array_transform_accumulation(self):
        """変形が累積的に適用されることのテスト"""
        array = Array()
        line = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
        
        # 90度ずつ回転させながら配列生成
        result = array.apply(
            [line],
            n_duplicates=0.4,
            offset=(0, 0, 0),
            rotate=(0.5, 0.5, 0.75),  # Z軸で90度回転
            center=(0, 0, 0)
        )
        
        assert len(result) == 4
        
        # 各複製が異なる向きを持つことを確認
        directions = []
        for vertices in result:
            direction = vertices[1] - vertices[0]
            directions.append(direction / np.linalg.norm(direction))
        
        # すべての方向が異なることを確認
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                assert not np.allclose(directions[i], directions[j], atol=0.1)

    def test_array_with_grid_shape(self):
        """グリッド形状での配列生成テスト"""
        array = Array()
        grid_shape = grid((0.5, 0.5), scale=(50, 50, 0))
        
        # グリッドは複数の線分を含むため、元の線分数を記録
        original_line_count = len(grid_shape)
        
        result = array.apply(
            grid_shape,
            n_duplicates=0.2,
            offset=(60, 60, 0),
            scale=(0.7, 0.7, 0.7)
        )
        
        # 元の線分数 * 複製数(2) = 結果の線分数
        assert len(result) == original_line_count * 2

    def test_array_parameter_types(self):
        """パラメータの型に関するテスト"""
        array = Array()
        square = polygon(4, scale=(30, 30, 0))
        
        # タプルの代わりにリストを使用
        result = array.apply(
            square,
            n_duplicates=0.1,
            offset=[10, 20, 30],
            rotate=[0.5, 0.5, 0.5],
            scale=[1.0, 1.0, 1.0],
            center=[0, 0, 0]
        )
        
        assert len(result) == 1

    def test_single_point_array(self):
        """単一の点を含む配列の処理テスト"""
        array = Array()
        single_point = np.array([[10, 20, 30]], dtype=np.float32)
        square = polygon(4, scale=(30, 30, 0))
        
        result = array.apply(
            [single_point, square[0]],
            n_duplicates=0.2,
            offset=(40, 0, 0)
        )
        
        # 複製数2で、各形状が2個ずつ生成される
        assert len(result) == 4  # 2形状 * 2複製 = 4
        
        # 頂点数でカウント
        single_point_count = sum(1 for vertices in result if len(vertices) == 1)
        square_count = sum(1 for vertices in result if len(vertices) == 5)  # polygonは閉じているので+1
        
        assert single_point_count == 2  # 単一点の2個の複製
        assert square_count == 2  # squareの2個の複製

    def test_fractional_duplicates(self):
        """小数点を含む複製数のテスト"""
        array = Array()
        square = polygon(4, scale=(30, 30, 0))
        
        # 0.25 * 10 = 2.5 → 2個の複製
        result = array.apply(square, n_duplicates=0.25, offset=(40, 0, 0))
        assert len(result) == 2
        
        # 0.35 * 10 = 3.5 → 3個の複製
        result = array.apply(square, n_duplicates=0.35, offset=(40, 0, 0))
        assert len(result) == 3