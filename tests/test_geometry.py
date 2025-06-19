"""util/geometry.pyのテストモジュール。"""

import numpy as np
import pytest
from util.geometry import transform_to_xy_plane, transform_back


class TestTransformToXYPlane:
    """transform_to_xy_plane関数のテストクラス。"""
    
    def test_already_on_xy_plane(self):
        """すでにXY平面上にある頂点の場合。"""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        # 変換後もXY平面上にあることを確認
        np.testing.assert_array_almost_equal(transformed[:, 2], 0)
        # 回転行列は単位行列
        np.testing.assert_array_almost_equal(R, np.eye(3))
        # z_offsetは0
        assert z_offset == 0.0
    
    def test_triangle_parallel_to_xy_plane(self):
        """XY平面に平行な三角形の場合。"""
        vertices = np.array([
            [0, 0, 5],
            [1, 0, 5],
            [0, 1, 5]
        ])
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        # z座標が0になることを確認
        np.testing.assert_array_almost_equal(transformed[:, 2], 0)
        # x, y座標は変わらない
        np.testing.assert_array_almost_equal(transformed[:, :2], vertices[:, :2])
        # z_offsetは5
        assert z_offset == 5.0
    
    def test_tilted_triangle(self):
        """傾いた三角形の場合。"""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1]
        ])
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        # すべてのz座標が0になることを確認
        np.testing.assert_array_almost_equal(transformed[:, 2], 0)
        # 回転行列が直交行列であることを確認
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        # 行列式が1（回転のみで反転なし）
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0)
    
    def test_less_than_three_vertices(self):
        """頂点が3つ未満の場合。"""
        # 1つの頂点
        vertices = np.array([[1, 2, 3]])
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        np.testing.assert_array_equal(transformed, vertices)
        np.testing.assert_array_equal(R, np.eye(3))
        assert z_offset == 0.0
        
        # 2つの頂点
        vertices = np.array([[1, 2, 3], [4, 5, 6]])
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        np.testing.assert_array_equal(transformed, vertices)
        np.testing.assert_array_equal(R, np.eye(3))
        assert z_offset == 0.0
    
    def test_colinear_points(self):
        """共線上の点の場合（法線ベクトルが定義できない）。"""
        vertices = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2]
        ])
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        # 元の頂点がそのまま返される
        np.testing.assert_array_equal(transformed, vertices)
        np.testing.assert_array_equal(R, np.eye(3))
        assert z_offset == 0.0
    
    def test_perpendicular_to_xy_plane(self):
        """XY平面に垂直な三角形の場合。"""
        vertices = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        
        # すべてのz座標が0になることを確認
        np.testing.assert_array_almost_equal(transformed[:, 2], 0)
        # 回転行列が直交行列であることを確認
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))


class TestTransformBack:
    """transform_back関数のテストクラス。"""
    
    def test_inverse_operation(self):
        """transform_to_xy_planeとtransform_backが逆操作であることの確認。"""
        # 様々な向きの三角形でテスト
        test_cases = [
            # 傾いた三角形
            np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]]),
            # XY平面に平行
            np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]]),
            # ランダムな三角形
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]]),
            # XZ平面上の三角形
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        ]
        
        for original_vertices in test_cases:
            # XY平面に変換
            transformed, R, z_offset = transform_to_xy_plane(original_vertices)
            
            # 元に戻す
            restored = transform_back(transformed, R, z_offset)
            
            # 元の頂点と一致することを確認
            np.testing.assert_array_almost_equal(restored, original_vertices)
    
    def test_identity_case(self):
        """単位行列とゼロオフセットの場合。"""
        vertices = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [5, 6, 0]
        ], dtype=np.float64)
        
        restored = transform_back(vertices, np.eye(3), 0.0)
        
        # 変化しないことを確認
        np.testing.assert_array_equal(restored, vertices)
    
    def test_z_offset_only(self):
        """z方向のオフセットのみの場合。"""
        vertices = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [5, 6, 0]
        ], dtype=np.float64)
        z_offset = 10.0
        
        restored = transform_back(vertices, np.eye(3), z_offset)
        
        # z座標のみ変化
        expected = vertices.copy()
        expected[:, 2] += z_offset
        np.testing.assert_array_equal(restored, expected)
    
    def test_rotation_preservation(self):
        """回転による距離と角度の保存を確認。"""
        # 正三角形を作成
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3)/2, 0]
        ])
        
        # 任意の回転行列を作成（z軸周りに45度回転）
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        z_offset = 5.0
        
        # 変換を適用
        transformed = vertices.copy()
        transformed[:, 2] += z_offset
        transformed = transformed @ R.T
        
        # transform_backで元に戻す
        restored = transform_back(vertices, R, z_offset)
        
        # 辺の長さが保存されることを確認
        for i in range(3):
            for j in range(i + 1, 3):
                original_dist = np.linalg.norm(vertices[i] - vertices[j])
                restored_dist = np.linalg.norm(restored[i] - restored[j])
                np.testing.assert_almost_equal(original_dist, restored_dist)


class TestIntegration:
    """統合テスト。"""
    
    def test_complex_polygon(self):
        """複雑な多角形での動作確認。"""
        # 五角形を作成（平面上にある点を生成）
        n = 5
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        # 傾いた平面上の五角形を作成
        # 最初はXY平面上に作成
        xy_vertices = np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros(n)
        ])
        
        # 任意の回転を適用して傾いた平面にする
        angle_x = np.pi / 6  # 30度
        angle_y = np.pi / 4  # 45度
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        
        # 回転とオフセットを適用
        vertices = xy_vertices @ Rx.T @ Ry.T
        vertices[:, 2] += 3.0  # z方向にオフセット
        
        # 変換と逆変換
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        restored = transform_back(transformed, R, z_offset)
        
        # 元に戻ることを確認
        np.testing.assert_array_almost_equal(restored, vertices, decimal=10)
        
        # XY平面上にあることを確認（数値誤差を考慮）
        np.testing.assert_allclose(transformed[:, 2], 0, atol=1e-10)
    
    def test_numerical_stability(self):
        """数値的安定性のテスト。"""
        # 非常に小さい値のテスト
        # 平面上の三角形を作成（数値誤差を避けるため）
        vertices = np.array([
            [1e-10, 0, 0],
            [0, 1e-10, 0],
            [0.5e-10, 0.5e-10, 0]
        ], dtype=np.float64)
        
        # 少し傾ける
        vertices[2, 2] = 1e-10
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        restored = transform_back(transformed, R, z_offset)
        
        # 相対誤差で比較
        np.testing.assert_allclose(restored, vertices, rtol=1e-8, atol=1e-25)
        
        # 大きい値のテスト（平面上の三角形）
        vertices = np.array([
            [1e6, 0, 0],
            [0, 1e6, 0],
            [0.5e6, 0.5e6, 1e6]
        ], dtype=np.float64)
        
        transformed, R, z_offset = transform_to_xy_plane(vertices)
        restored = transform_back(transformed, R, z_offset)
        
        np.testing.assert_allclose(restored, vertices, rtol=1e-10, atol=1e-6)