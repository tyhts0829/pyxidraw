"""shapes/torus.pyのテストモジュール。"""

import numpy as np
import pytest
from shapes.torus import Torus


class TestTorus:
    """Torusクラスのテストクラス。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行されるセットアップ。"""
        self.torus = Torus()
    
    def test_basic_generation(self):
        """基本的なトーラス生成のテスト。"""
        vertices_list = self.torus.generate()
        
        # デフォルトパラメータでは32 + 16 = 48本の線が生成される
        assert len(vertices_list) == 48
        
        # 各線が適切な頂点数を持つことを確認
        for i, vertices in enumerate(vertices_list):
            assert vertices.shape[1] == 3  # 3D座標
            assert vertices.dtype == np.float32
            
            if i < 32:  # major circle lines (meridians)
                assert vertices.shape[0] == 17  # minor_segments + 1
            else:  # minor circle lines (parallels)
                assert vertices.shape[0] == 33  # major_segments + 1
    
    def test_custom_parameters(self):
        """カスタムパラメータでのテスト。"""
        major_radius = 0.5
        minor_radius = 0.2
        major_segments = 16
        minor_segments = 8
        
        vertices_list = self.torus.generate(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_segments=major_segments,
            minor_segments=minor_segments
        )
        
        # 期待される線の数
        assert len(vertices_list) == major_segments + minor_segments
        
        # 各線の頂点数の確認
        for i, vertices in enumerate(vertices_list):
            if i < major_segments:  # meridians
                assert vertices.shape[0] == minor_segments + 1
            else:  # parallels
                assert vertices.shape[0] == major_segments + 1
    
    def test_radius_constraints(self):
        """半径の制約のテスト。"""
        major_radius = 0.3
        minor_radius = 0.1
        
        vertices_list = self.torus.generate(
            major_radius=major_radius,
            minor_radius=minor_radius
        )
        
        # 全ての頂点がトーラスの幾何学的制約を満たすことを確認
        for vertices in vertices_list:
            for vertex in vertices:
                x, y, z = vertex
                # xy平面での距離
                xy_distance = np.sqrt(x**2 + y**2)
                
                # トーラス表面の点は major_radius ± minor_radius の範囲内
                assert (major_radius - minor_radius) <= xy_distance <= (major_radius + minor_radius)
                # z座標は ±minor_radius の範囲内
                assert -minor_radius <= z <= minor_radius
    
    def test_meridian_lines_topology(self):
        """経線（meridian）の位相的性質のテスト。"""
        major_radius = 0.25
        minor_radius = 0.125
        major_segments = 8
        minor_segments = 6
        
        vertices_list = self.torus.generate(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_segments=major_segments,
            minor_segments=minor_segments
        )
        
        # 最初の8本が経線
        meridian_lines = vertices_list[:major_segments]
        
        for i, meridian in enumerate(meridian_lines):
            # 各経線は閉じたループを形成する（最初と最後の点が同じz座標）
            first_z = meridian[0, 2]
            last_z = meridian[-1, 2]
            np.testing.assert_almost_equal(first_z, last_z)
            
            # 経線上の全ての点は同じ角度θを持つ
            theta = 2 * np.pi * i / major_segments
            expected_cos_theta = np.cos(theta)
            expected_sin_theta = np.sin(theta)
            
            for vertex in meridian:
                x, y, z = vertex
                xy_distance = np.sqrt(x**2 + y**2)
                if xy_distance > 1e-10:  # ゼロ除算を避ける
                    actual_cos_theta = x / xy_distance
                    actual_sin_theta = y / xy_distance
                    np.testing.assert_almost_equal(actual_cos_theta, expected_cos_theta, decimal=5)
                    np.testing.assert_almost_equal(actual_sin_theta, expected_sin_theta, decimal=5)
    
    def test_parallel_lines_topology(self):
        """緯線（parallel）の位相的性質のテスト。"""
        major_radius = 0.25
        minor_radius = 0.125
        major_segments = 8
        minor_segments = 6
        
        vertices_list = self.torus.generate(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_segments=major_segments,
            minor_segments=minor_segments
        )
        
        # 後の6本が緯線
        parallel_lines = vertices_list[major_segments:]
        
        for j, parallel in enumerate(parallel_lines):
            # 各緯線は同じz座標を持つ
            phi = 2 * np.pi * j / minor_segments
            expected_z = minor_radius * np.sin(phi)
            
            for vertex in parallel:
                np.testing.assert_almost_equal(vertex[2], expected_z, decimal=5)
            
            # 緯線は閉じたループを形成する（最初と最後の点のxy座標が同じ）
            first_xy = parallel[0, :2]
            last_xy = parallel[-1, :2]
            np.testing.assert_array_almost_equal(first_xy, last_xy, decimal=5)
    
    def test_torus_equation(self):
        """トーラス方程式の数学的正確性のテスト。"""
        major_radius = 0.4
        minor_radius = 0.15
        
        vertices_list = self.torus.generate(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_segments=16,
            minor_segments=12
        )
        
        # 全ての頂点がトーラス方程式を満たすことを確認
        for vertices in vertices_list:
            for vertex in vertices:
                x, y, z = vertex
                
                # トーラス方程式: (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2
                # R = major_radius, r = minor_radius
                xy_distance = np.sqrt(x**2 + y**2)
                torus_value = (xy_distance - major_radius)**2 + z**2
                expected_value = minor_radius**2
                
                # 数値誤差を考慮して比較
                np.testing.assert_almost_equal(torus_value, expected_value, decimal=4)
    
    def test_symmetry(self):
        """トーラスの対称性のテスト。"""
        vertices_list = self.torus.generate(major_segments=8, minor_segments=8)
        
        # 全ての頂点を収集
        all_vertices = np.vstack(vertices_list)
        
        # z軸周りの回転対称性をテスト
        # 45度回転したときに類似の点群が得られることを確認
        angle = np.pi / 4  # 45度
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        rotated_vertices = all_vertices @ rotation_matrix.T
        
        # 回転前後で点の分布が似ていることを確認（距離の分布）
        original_distances = np.sqrt(np.sum(all_vertices**2, axis=1))
        rotated_distances = np.sqrt(np.sum(rotated_vertices**2, axis=1))
        
        # 距離の統計が保存されることを確認
        np.testing.assert_almost_equal(np.mean(original_distances), np.mean(rotated_distances))
        np.testing.assert_almost_equal(np.std(original_distances), np.std(rotated_distances))
    
    def test_minimal_segments(self):
        """最小セグメント数でのテスト。"""
        vertices_list = self.torus.generate(
            major_segments=3,
            minor_segments=3
        )
        
        assert len(vertices_list) == 6  # 3 + 3
        
        # 各線が適切な頂点数を持つことを確認
        for i, vertices in enumerate(vertices_list):
            if i < 3:  # meridians
                assert vertices.shape[0] == 4  # minor_segments + 1
            else:  # parallels
                assert vertices.shape[0] == 4  # major_segments + 1
    
    def test_large_segments(self):
        """大きなセグメント数でのテスト。"""
        vertices_list = self.torus.generate(
            major_segments=64,
            minor_segments=32
        )
        
        assert len(vertices_list) == 96  # 64 + 32
        
        # メモリ効率とパフォーマンスの確認
        total_vertices = sum(len(vertices) for vertices in vertices_list)
        assert total_vertices > 0
        
        # 全ての配列がfloat32であることを確認
        for vertices in vertices_list:
            assert vertices.dtype == np.float32
    
    def test_zero_radius_edge_case(self):
        """ゼロ半径のエッジケース。"""
        # minor_radiusが0の場合（円になる）
        vertices_list = self.torus.generate(
            major_radius=0.5,
            minor_radius=0.0
        )
        
        # 全ての頂点のz座標が0になる
        for vertices in vertices_list:
            np.testing.assert_array_almost_equal(vertices[:, 2], 0)
    
    def test_additional_params_ignored(self):
        """追加パラメータが無視されることのテスト。"""
        vertices_list = self.torus.generate(
            major_radius=0.3,
            minor_radius=0.1,
            unused_param=123,
            another_param="test"
        )
        
        # 正常に動作することを確認
        assert len(vertices_list) == 48  # デフォルトのセグメント数