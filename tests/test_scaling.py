import numpy as np
import pytest

from effects.scaling import Scaling
from api.shapes import polygon


class TestScaling:
    """Scalingエフェクトのテストクラス"""

    def test_basic_scaling(self):
        """基本的なスケーリングのテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # 2倍にスケール
        scaled = scaling.apply(square, scale=(2, 2, 2))
        
        # 頂点数が変わらないことを確認
        assert len(scaled[0]) == len(square[0])
        
        # スケールが適用されていることを確認（原点基準）
        assert np.allclose(scaled[0][0], square[0][0] * 2)

    def test_scaling_with_center(self):
        """中心点を指定したスケーリングのテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # 中心点(50, 50, 0)で0.5倍にスケール
        scaled = scaling.apply(square, center=(50, 50, 0), scale=(0.5, 0.5, 0.5))
        
        # 頂点数が変わらないことを確認
        assert len(scaled[0]) == len(square[0])
        
        # 中心点からの距離が半分になっていることを確認
        center = np.array([50, 50, 0])
        original_dist = np.linalg.norm(square[0][0] - center)
        scaled_dist = np.linalg.norm(scaled[0][0] - center)
        assert np.isclose(scaled_dist, original_dist * 0.5)

    def test_identity_scaling(self):
        """スケール係数が1の場合のテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # スケールなし
        scaled = scaling.apply(square, scale=(1, 1, 1))
        
        # 元のデータと同じ値（コピー）であることを確認
        assert np.array_equal(square[0], scaled[0])
        assert square[0] is not scaled[0]  # 別のオブジェクトであることを確認

    def test_empty_list(self):
        """空のリストの処理テスト"""
        scaling = Scaling()
        result = scaling.apply([])
        assert result == []

    def test_non_uniform_scaling(self):
        """非均一スケーリングのテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # X軸2倍、Y軸0.5倍、Z軸1倍
        scaled = scaling.apply(square, scale=(2, 0.5, 1))
        
        # 頂点数が変わらないことを確認
        assert len(scaled[0]) == len(square[0])
        
        # 各軸で正しくスケールされていることを確認
        assert np.isclose(scaled[0][0][0], square[0][0][0] * 2)  # X軸
        assert np.isclose(scaled[0][0][1], square[0][0][1] * 0.5)  # Y軸
        assert np.isclose(scaled[0][0][2], square[0][0][2] * 1)  # Z軸

    def test_multiple_shapes(self):
        """複数の形状のスケーリングテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        triangle = polygon(3, scale=(30, 30, 0))
        
        # 複数の形状を同時にスケール
        scaled = scaling.apply([square[0], triangle[0]], scale=(1.5, 1.5, 1.5))
        
        assert len(scaled) == 2
        assert len(scaled[0]) == len(square[0])
        assert len(scaled[1]) == len(triangle[0])

    def test_negative_scaling(self):
        """負のスケール係数のテスト（反転）"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # X軸で反転
        scaled = scaling.apply(square, scale=(-1, 1, 1))
        
        # 頂点数が変わらないことを確認
        assert len(scaled[0]) == len(square[0])
        
        # X座標が反転していることを確認
        assert np.isclose(scaled[0][0][0], -square[0][0][0])
        assert np.isclose(scaled[0][0][1], square[0][0][1])

    def test_zero_scaling(self):
        """ゼロスケーリングのテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # Y軸を0にスケール（平面に潰す）
        scaled = scaling.apply(square, scale=(1, 0, 1))
        
        # すべての頂点のY座標が0になることを確認
        assert np.allclose(scaled[0][:, 1], 0)

    def test_empty_vertices_in_list(self):
        """空の頂点配列を含むリストの処理テスト"""
        scaling = Scaling()
        empty_array = np.array([]).reshape(0, 3)
        square = polygon(4, scale=(50, 50, 0))
        
        result = scaling.apply([empty_array, square[0]], scale=(2, 2, 2))
        
        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == len(square[0])

    def test_scaling_preserves_relative_positions(self):
        """スケーリングが相対位置を保持することをテスト"""
        scaling = Scaling()
        square = polygon(4, scale=(50, 50, 0))
        
        # 元の形状の重心を計算
        centroid = np.mean(square[0], axis=0)
        
        # スケール後の重心
        scaled = scaling.apply(square, scale=(3, 3, 3))
        scaled_centroid = np.mean(scaled[0], axis=0)
        
        # 重心も同じ比率でスケールされることを確認
        assert np.allclose(scaled_centroid, centroid * 3)