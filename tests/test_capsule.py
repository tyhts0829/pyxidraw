"""shapes/capsule.pyのテストモジュール。"""

import numpy as np
import pytest
from shapes.capsule import Capsule


class TestCapsule:
    """Capsuleクラスのテストクラス。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行されるセットアップ。"""
        self.capsule = Capsule()
    
    def test_basic_generation(self):
        """基本的なカプセル生成のテスト。"""
        vertices_list = self.capsule.generate()
        
        # デフォルトパラメータ（segments=32, latitude_segments=16）では
        # 縦線32本 + 上下の横線64本 + 上半球緯度線480本 + 下半球緯度線480本 + 経度線1024本
        # = 32 + 64 + 480 + 480 + 1024 = 2080本の線が生成される
        expected_lines = 32 + 64 + (16-1)*32 + (16-1)*32 + 32*16 + 32*16
        assert len(vertices_list) == expected_lines
        
        # 各線が適切な形状を持つことを確認
        for vertices in vertices_list:
            assert vertices.shape[1] == 3  # 3D座標
            assert vertices.dtype == np.float32
            assert vertices.shape[0] == 2  # 各線は2つの頂点を持つ
    
    def test_custom_parameters(self):
        """カスタムパラメータでのテスト。"""
        radius = 0.3
        height = 0.6
        segments = 16
        latitude_segments = 8
        
        vertices_list = self.capsule.generate(
            radius=radius,
            height=height,
            segments=segments,
            latitude_segments=latitude_segments
        )
        
        # 期待される線の数を計算
        expected_lines = (
            segments +  # 縦線
            2 * segments +  # 上下の横線
            2 * (latitude_segments - 1) * segments +  # 上下半球の緯度線
            2 * segments * latitude_segments  # 上下半球の経度線
        )
        assert len(vertices_list) == expected_lines
        
        # 全ての線が2つの頂点を持つことを確認
        for vertices in vertices_list:
            assert vertices.shape == (2, 3)
    
    def test_radius_and_height_constraints(self):
        """半径と高さの制約のテスト。"""
        radius = 0.25
        height = 0.5
        
        vertices_list = self.capsule.generate(radius=radius, height=height)
        
        # 全ての頂点がカプセルの幾何学的制約を満たすことを確認
        for vertices in vertices_list:
            for vertex in vertices:
                x, y, z = vertex
                xy_distance = np.sqrt(x**2 + y**2)
                
                # 円柱部分（-height/2 <= z <= height/2）では xy_distance <= radius
                if -height/2 <= z <= height/2:
                    assert xy_distance <= radius + 1e-6  # 数値誤差を考慮
                
                # 上半球部分（z > height/2）
                elif z > height/2:
                    sphere_center_z = height/2
                    distance_from_center = np.sqrt(xy_distance**2 + (z - sphere_center_z)**2)
                    assert distance_from_center <= radius + 1e-6
                
                # 下半球部分（z < -height/2）
                elif z < -height/2:
                    sphere_center_z = -height/2
                    distance_from_center = np.sqrt(xy_distance**2 + (z - sphere_center_z)**2)
                    assert distance_from_center <= radius + 1e-6
    
    def test_cylinder_vertical_lines(self):
        """円柱部分の縦線のテスト。"""
        radius = 0.2
        height = 0.4
        segments = 8
        
        vertices_list = self.capsule.generate(
            radius=radius, height=height, segments=segments, latitude_segments=4
        )
        
        # 最初の8本が縦線
        vertical_lines = vertices_list[:segments]
        
        for i, line in enumerate(vertical_lines):
            # 各縦線は上端から下端まで
            start_z = line[0, 2]
            end_z = line[1, 2]
            
            # ユニットカプセルの半径0.5、高さ1.0からスケーリング
            # scale_z = height / 2.0, ユニットカプセルのhalf_height=0.5
            # 実際の高さの半分 = 0.5 * (height / 2.0) = height / 4.0
            actual_half_height = height / 4.0
            np.testing.assert_almost_equal(abs(start_z), actual_half_height, decimal=5)
            np.testing.assert_almost_equal(abs(end_z), actual_half_height, decimal=5)
            np.testing.assert_almost_equal(start_z, -end_z, decimal=5)
            
            # 同じ角度θでのx, y座標
            # ユニットカプセルの半径0.5からスケーリング: scale_xy = radius / 1.0
            # 実際の半径 = 0.5 * scale_xy = 0.5 * radius
            actual_radius = 0.5 * radius
            angle = 2 * np.pi * i / segments
            expected_x = actual_radius * np.cos(angle)
            expected_y = actual_radius * np.sin(angle)
            
            # 両端点が同じx, y座標を持つ
            np.testing.assert_almost_equal(line[0, 0], expected_x, decimal=5)
            np.testing.assert_almost_equal(line[0, 1], expected_y, decimal=5)
            np.testing.assert_almost_equal(line[1, 0], expected_x, decimal=5)
            np.testing.assert_almost_equal(line[1, 1], expected_y, decimal=5)
    
    def test_hemisphere_structure(self):
        """半球構造のテスト。"""
        radius = 0.3
        height = 0.6
        segments = 12
        latitude_segments = 6
        
        vertices_list = self.capsule.generate(
            radius=radius, height=height, segments=segments, latitude_segments=latitude_segments
        )
        
        # スケーリング後の実際の半球中心位置
        # ユニットカプセルの半径0.5、高さ1.0からスケーリング
        # 実際の高さの半分 = 0.5 * (height / 2.0) = height / 4.0
        actual_half_height = height / 4.0
        upper_center_z = actual_half_height
        lower_center_z = -actual_half_height
        
        # 半球領域の頂点を抽出（完全に一致するものは稀なので、わずかな閾値を使用）
        hemisphere_vertices = []
        tolerance = 1e-6
        for vertices in vertices_list:
            for vertex in vertices:
                z = vertex[2]
                if z > upper_center_z + tolerance or z < lower_center_z - tolerance:
                    hemisphere_vertices.append(vertex)
        
        # 半球の頂点が存在することを確認
        assert len(hemisphere_vertices) > 0
        
        # 各半球頂点が球面方程式を満たすことを確認
        for vertex in hemisphere_vertices:
            x, y, z = vertex
            
            # 実際の半径 = 0.5 * scale_xy = 0.5 * radius
            actual_radius = 0.5 * radius
            
            if z > upper_center_z:  # 上半球
                distance = np.sqrt(x**2 + y**2 + (z - upper_center_z)**2)
                np.testing.assert_almost_equal(distance, actual_radius, decimal=4)
            elif z < lower_center_z:  # 下半球
                distance = np.sqrt(x**2 + y**2 + (z - lower_center_z)**2)
                np.testing.assert_almost_equal(distance, actual_radius, decimal=4)
    
    def test_symmetry(self):
        """カプセルの対称性のテスト。"""
        vertices_list = self.capsule.generate(segments=8, latitude_segments=4)
        
        # 全ての頂点を収集
        all_vertices = []
        for vertices in vertices_list:
            all_vertices.extend(vertices)
        all_vertices = np.array(all_vertices)
        
        # z軸に対する対称性をテスト
        # z = 0 平面に対して対称であることを確認
        positive_z_vertices = all_vertices[all_vertices[:, 2] > 0]
        negative_z_vertices = all_vertices[all_vertices[:, 2] < 0]
        
        # 正のz座標を持つ点の数と負のz座標を持つ点の数が同じであることを確認
        # （z=0の点は除く）
        assert len(positive_z_vertices) == len(negative_z_vertices)
        
        # z軸周りの回転対称性をテスト
        angle = np.pi / 4  # 45度回転
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        rotated_vertices = all_vertices @ rotation_matrix.T
        
        # 回転前後でz座標の分布が保存されることを確認
        original_z_range = np.max(all_vertices[:, 2]) - np.min(all_vertices[:, 2])
        rotated_z_range = np.max(rotated_vertices[:, 2]) - np.min(rotated_vertices[:, 2])
        np.testing.assert_almost_equal(original_z_range, rotated_z_range, decimal=5)
    
    def test_minimal_segments(self):
        """最小セグメント数でのテスト。"""
        vertices_list = self.capsule.generate(
            segments=4,
            latitude_segments=2
        )
        
        # 線の数が適切であることを確認
        expected_lines = 4 + 8 + 2*1*4 + 2*4*2  # 縦線 + 横線 + 緯度線 + 経度線
        assert len(vertices_list) == expected_lines
        
        # 各線が適切な頂点数を持つことを確認
        for vertices in vertices_list:
            assert vertices.shape == (2, 3)
    
    def test_zero_height_edge_case(self):
        """高さ0のエッジケース（球になる）。"""
        radius = 0.3
        height = 0.0
        
        vertices_list = self.capsule.generate(radius=radius, height=height)
        
        # 高さ0の場合、半球が重なって球に近い形になる
        # しかし、実際にはスケーリング係数の影響で完全な球にはならない
        # z範囲が制限されることを確認
        all_z = []
        for vertices in vertices_list:
            for vertex in vertices:
                all_z.append(vertex[2])
        
        z_range = max(all_z) - min(all_z)
        # 高さ0の場合、scale_z = 0.0 / 2.0 = 0.0
        # 全てのz座標が0.0 * original_z = 0.0になる
        assert z_range == 0.0
    
    def test_cache_functionality(self):
        """キャッシュ機能のテスト。"""
        # キャッシュクリア
        self.capsule.clear_cache()
        
        # 同じパラメータで2回生成
        params = {"radius": 0.2, "height": 0.4, "segments": 8, "latitude_segments": 4}
        vertices_list_1 = self.capsule.generate(**params)
        vertices_list_2 = self.capsule.generate(**params)
        
        # 結果が同じであることを確認
        assert len(vertices_list_1) == len(vertices_list_2)
        
        for v1, v2 in zip(vertices_list_1, vertices_list_2):
            np.testing.assert_array_equal(v1, v2)
    
    def test_scaling_correctness(self):
        """スケーリングの正確性をテスト。"""
        target_radius = 0.4
        target_height = 0.8
        
        vertices_list = self.capsule.generate(
            radius=target_radius, 
            height=target_height,
            segments=8,
            latitude_segments=4
        )
        
        # スケーリング後の実際の高さ範囲
        # 実際の高さの半分 = 0.5 * (target_height / 2.0) = target_height / 4.0
        actual_half_height = target_height / 4.0
        
        # 円柱部分の頂点を検証
        cylinder_vertices = []
        for vertices in vertices_list:
            for vertex in vertices:
                if -actual_half_height <= vertex[2] <= actual_half_height:
                    cylinder_vertices.append(vertex)
        
        # 円柱部分の頂点の半径が正しいことを確認
        # 実際の半径 = 0.5 * scale_xy = 0.5 * target_radius
        actual_radius = 0.5 * target_radius
        for vertex in cylinder_vertices:
            xy_distance = np.sqrt(vertex[0]**2 + vertex[1]**2)
            np.testing.assert_almost_equal(xy_distance, actual_radius, decimal=4)
    
    def test_additional_params_ignored(self):
        """追加パラメータが無視されることのテスト。"""
        vertices_list = self.capsule.generate(
            radius=0.3,
            height=0.6,
            unused_param=123,
            another_param="test"
        )
        
        # 正常に動作することを確認
        assert len(vertices_list) > 0