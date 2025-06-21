import numpy as np
import pytest

from effects.transform import Transform
from api.shapes import polygon


class TestTransform:
    """Transformエフェクトのテストクラス"""

    def test_basic_transform(self):
        """基本的な変換のテスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        
        # 移動、スケール、回転を同時に適用
        transformed = transform.apply(
            square,
            center=(10, 10, 0),
            scale=(2, 2, 1),
            rotate=(0, 0, np.pi/4)
        )
        
        # 頂点数が変わらないことを確認
        assert len(transformed[0]) == len(square[0])
        
        # 変換が適用されていることを確認
        assert not np.allclose(square[0][0], transformed[0][0])

    def test_center_only(self):
        """中心点の移動のみのテスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        
        # 移動のみ
        transformed = transform.apply(
            square,
            center=(100, 100, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, 0)
        )
        
        # 頂点数が変わらないことを確認
        assert len(transformed[0]) == len(square[0])
        
        # 移動が正しく適用されていることを確認
        offset = transformed[0][0] - square[0][0]
        assert np.allclose(offset, [100, 100, 0])

    def test_scale_only(self):
        """スケールのみのテスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        
        # スケールのみ
        transformed = transform.apply(
            square,
            center=(0, 0, 0),
            scale=(3, 3, 3),
            rotate=(0, 0, 0)
        )
        
        # スケールが正しく適用されていることを確認
        assert np.allclose(transformed[0][0], square[0][0] * 3)

    def test_rotate_only(self):
        """回転のみのテスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        
        # 回転のみ
        transformed = transform.apply(
            square,
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, np.pi/2)
        )
        
        # 頂点数が変わらないことを確認
        assert len(transformed[0]) == len(square[0])
        
        # 回転が適用されていることを確認（距離は保持）
        original_dist = np.linalg.norm(square[0][0])
        transformed_dist = np.linalg.norm(transformed[0][0])
        assert np.isclose(original_dist, transformed_dist)

    def test_identity_transform(self):
        """恒等変換のテスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        
        # 恒等変換
        transformed = transform.apply(
            square,
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, 0)
        )
        
        # 元のデータと同じであることを確認
        assert np.array_equal(square[0], transformed[0])

    def test_empty_list(self):
        """空のリストの処理テスト"""
        transform = Transform()
        result = transform.apply([])
        assert result == []

    def test_multiple_shapes(self):
        """複数の形状の変換テスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        triangle = polygon(3, scale=(30, 30, 0))
        
        # 複数の形状を同時に変換
        transformed = transform.apply(
            [square[0], triangle[0]],
            center=(20, 20, 0),
            scale=(1.5, 1.5, 1),
            rotate=(np.pi/6, 0, 0)
        )
        
        assert len(transformed) == 2
        assert len(transformed[0]) == len(square[0])
        assert len(transformed[1]) == len(triangle[0])

    def test_combined_transform_order(self):
        """変換の順序のテスト（回転→スケール→移動）"""
        transform = Transform()
        square = polygon(4, scale=(10, 10, 0))
        
        # 複合変換
        transformed = transform.apply(
            square,
            center=(50, 50, 0),
            scale=(2, 2, 1),
            rotate=(0, 0, np.pi/2)
        )
        
        # 手動で同じ変換を計算
        # 1. 回転
        cos_r = np.cos(np.pi/2)
        sin_r = np.sin(np.pi/2)
        rotated = square[0].copy()
        for i in range(len(rotated)):
            x, y = rotated[i][0], rotated[i][1]
            rotated[i][0] = x * cos_r - y * sin_r
            rotated[i][1] = x * sin_r + y * cos_r
        
        # 2. スケール
        scaled = rotated * np.array([2, 2, 1])
        
        # 3. 移動
        moved = scaled + np.array([50, 50, 0])
        
        # 結果が一致することを確認
        assert np.allclose(transformed[0], moved)

    def test_3d_transform(self):
        """3D変換のテスト"""
        transform = Transform()
        # 3D座標を持つ形状を作成
        vertices = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)
        
        # 3軸すべてで変換
        transformed = transform.apply(
            [vertices],
            center=(5, 5, 5),
            scale=(2, 3, 4),
            rotate=(np.pi/4, np.pi/3, np.pi/6)
        )
        
        # 頂点数が変わらないことを確認
        assert len(transformed[0]) == len(vertices)

    def test_non_uniform_scale_with_rotation(self):
        """非均一スケールと回転の組み合わせテスト"""
        transform = Transform()
        square = polygon(4, scale=(50, 50, 0))
        
        # 非均一スケールと回転
        transformed = transform.apply(
            square,
            center=(0, 0, 0),
            scale=(2, 0.5, 1),
            rotate=(0, 0, np.pi/4)
        )
        
        # 頂点数が変わらないことを確認
        assert len(transformed[0]) == len(square[0])
        
        # 変換が適用されていることを確認
        assert not np.allclose(square[0][0], transformed[0][0])