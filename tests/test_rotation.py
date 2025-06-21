import numpy as np
import pytest

from effects.rotation import Rotation
from api.shapes import polygon


class TestRotation:
    """Rotationエフェクトのテストクラス"""

    def test_basic_rotation(self):
        """基本的な回転のテスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        
        # Z軸周りに45度回転
        rotated = rotation.apply(square, rotate=(0, 0, np.pi/4))
        
        # 頂点数が変わらないことを確認
        assert len(rotated[0]) == len(square[0])
        
        # 回転が適用されていることを確認（元の位置と異なる）
        assert not np.allclose(square[0][0], rotated[0][0])

    def test_rotation_with_center(self):
        """中心点を指定した回転のテスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        
        # 中心点(50, 50, 0)で90度回転
        rotated = rotation.apply(square, center=(50, 50, 0), rotate=(0, 0, np.pi/2))
        
        # 頂点数が変わらないことを確認
        assert len(rotated[0]) == len(square[0])

    def test_identity_rotation(self):
        """回転角度がゼロの場合のテスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        
        # 回転なし
        rotated = rotation.apply(square, rotate=(0, 0, 0))
        
        # 元のデータと同じ値（コピー）であることを確認
        assert np.array_equal(square[0], rotated[0])
        assert square[0] is not rotated[0]  # 別のオブジェクトであることを確認

    def test_empty_list(self):
        """空のリストの処理テスト"""
        rotation = Rotation()
        result = rotation.apply([])
        assert result == []

    def test_multiple_shapes(self):
        """複数の形状の回転テスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        triangle = polygon(3, scale=(30, 30, 0))
        
        # 複数の形状を同時に回転
        rotated = rotation.apply([square[0], triangle[0]], rotate=(np.pi/6, 0, 0))
        
        assert len(rotated) == 2
        assert len(rotated[0]) == len(square[0])
        assert len(rotated[1]) == len(triangle[0])

    def test_3d_rotation(self):
        """3軸同時回転のテスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        
        # 3軸すべてで回転
        rotated = rotation.apply(square, rotate=(np.pi/4, np.pi/3, np.pi/6))
        
        # 頂点数が変わらないことを確認
        assert len(rotated[0]) == len(square[0])
        
        # 回転が適用されていることを確認
        assert not np.allclose(square[0][0], rotated[0][0])

    def test_rotation_preserves_shape_size(self):
        """回転が形状のサイズを保持することをテスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        
        # 元の形状の中心からの距離を計算
        center = np.mean(square[0], axis=0)
        original_distances = np.linalg.norm(square[0] - center, axis=1)
        
        # 回転後
        rotated = rotation.apply(square, rotate=(0, 0, np.pi/3))
        rotated_center = np.mean(rotated[0], axis=0)
        rotated_distances = np.linalg.norm(rotated[0] - rotated_center, axis=1)
        
        # 距離が保持されていることを確認（浮動小数点誤差を考慮）
        assert np.allclose(original_distances, rotated_distances, rtol=1e-5)

    def test_empty_vertices_in_list(self):
        """空の頂点配列を含むリストの処理テスト"""
        rotation = Rotation()
        empty_array = np.array([]).reshape(0, 3)
        square = polygon(4, scale=(50, 50, 0))
        
        result = rotation.apply([empty_array, square[0]], rotate=(0, 0, np.pi/4))
        
        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == len(square[0])

    def test_rotation_angles_precision(self):
        """回転角度の精度テスト"""
        rotation = Rotation()
        square = polygon(4, scale=(50, 50, 0))
        
        # 極小の角度（1e-11ラジアン）
        tiny_rotation = rotation.apply(square, rotate=(1e-11, 1e-11, 1e-11))
        
        # ほぼ元のデータと同じであることを確認
        assert np.array_equal(square[0], tiny_rotation[0])