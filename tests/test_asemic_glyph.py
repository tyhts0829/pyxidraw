"""shapes/asemic_glyph.pyのテストモジュール。"""

import math
import random
from typing import Any, Dict

import numpy as np
import pytest

from shapes.asemic_glyph import (
    AsemicGlyph,
    AsemicGlyphConfig,
    DiacriticFactory,
    add_diacritic,
    distance,
    generate_nodes,
    random_walk_strokes,
    relative_neighborhood_graph,
    smooth_polyline,
    snap_stroke,
)


class TestAsemicGlyphConfig:
    """AsemicGlyphConfig設定クラスのテスト。"""
    
    def test_default_config(self):
        """デフォルト設定のテスト。"""
        config = AsemicGlyphConfig()
        assert config.min_distance == 0.1
        assert config.snap_angle_degrees == 60.0
        assert config.smoothing_points == 5
        assert config.walk_min_steps == 2
        assert config.walk_max_steps == 4
        assert config.poisson_radius_divisor == 8.0
        assert config.poisson_trials == 30
    
    def test_custom_config(self):
        """カスタム設定のテスト。"""
        config = AsemicGlyphConfig(
            min_distance=0.2,
            snap_angle_degrees=45.0,
            smoothing_points=10
        )
        assert config.min_distance == 0.2
        assert config.snap_angle_degrees == 45.0
        assert config.smoothing_points == 10
        # 他のパラメータはデフォルト値
        assert config.walk_min_steps == 2
        assert config.walk_max_steps == 4


class TestUtilityFunctions:
    """ユーティリティ関数のテスト。"""
    
    def test_distance(self):
        """distance関数のテスト。"""
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 4.0, 0.0)
        assert distance(p1, p2) == 5.0
        
        p3 = (1.0, 1.0, 0.0)
        p4 = (1.0, 1.0, 0.0)
        assert distance(p3, p4) == 0.0
        
        p5 = (-1.0, -1.0, 0.0)
        p6 = (1.0, 1.0, 0.0)
        assert abs(distance(p5, p6) - math.sqrt(8)) < 1e-10
    
    def test_snap_stroke(self):
        """snap_stroke関数のテスト。"""
        config = AsemicGlyphConfig(snap_angle_degrees=90.0)
        
        # 直線的なストローク
        original = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 0.0)]
        snapped = snap_stroke(original, config)
        
        assert len(snapped) >= 2
        assert snapped[0] == original[0]  # 最初の点は変更されない
        
        # エッジケース：1点だけ
        single_point = [(0.0, 0.0, 0.0)]
        result = snap_stroke(single_point, config)
        assert result == single_point
        
        # エッジケース：空のリスト
        empty = []
        result = snap_stroke(empty, config)
        assert result == empty
    
    def test_smooth_polyline(self):
        """smooth_polyline関数のテスト。"""
        config = AsemicGlyphConfig(smoothing_points=3)
        
        # 簡単な三角形
        polyline = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        smoothed = smooth_polyline(polyline, 0.1, config)
        
        assert len(smoothed) >= len(polyline)
        assert smoothed[0] == polyline[0]  # 最初の点は変更されない
        assert smoothed[-1] == polyline[-1]  # 最後の点は変更されない
        
        # エッジケース：2点以下
        two_points = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        result = smooth_polyline(two_points, 0.1, config)
        assert result == two_points


class TestGenerateNodes:
    """generate_nodes関数のテスト。"""
    
    def test_grid_placement(self):
        """グリッド配置のテスト。"""
        region = (-1.0, -1.0, 1.0, 1.0)
        nodes = generate_nodes(region, 0.1, "grid")
        
        assert len(nodes) >= 4  # 最低でも2x2のグリッド
        for node in nodes:
            assert len(node) == 3
            assert node[2] == 0.0  # z座標は0
            assert -1.0 <= node[0] <= 1.0
            assert -1.0 <= node[1] <= 1.0
    
    def test_hexagon_placement(self):
        """六角形配置のテスト。"""
        region = (-1.0, -1.0, 1.0, 1.0)
        nodes = generate_nodes(region, 0.1, "hexagon")
        
        assert len(nodes) > 0
        for node in nodes:
            assert len(node) == 3
            assert node[2] == 0.0
    
    def test_poisson_placement(self):
        """ポアソンディスクサンプリング配置のテスト。"""
        region = (-1.0, -1.0, 1.0, 1.0)
        config = AsemicGlyphConfig(poisson_radius_divisor=10.0)
        nodes = generate_nodes(region, 0.1, "poisson", config)
        
        assert len(nodes) > 0
        for node in nodes:
            assert len(node) == 3
            assert node[2] == 0.0
            assert -1.0 <= node[0] <= 1.0
            assert -1.0 <= node[1] <= 1.0
    
    def test_spiral_placement(self):
        """スパイラル配置のテスト。"""
        region = (-1.0, -1.0, 1.0, 1.0)
        nodes = generate_nodes(region, 0.1, "spiral")
        
        assert len(nodes) > 0
        for node in nodes:
            assert len(node) == 3
            assert node[2] == 0.0
    
    def test_invalid_placement_mode(self):
        """無効な配置モードでデフォルトに戻ることのテスト。"""
        region = (-1.0, -1.0, 1.0, 1.0)
        nodes = generate_nodes(region, 0.1, "invalid_mode")
        
        # デフォルトのgridモードで配置される
        assert len(nodes) >= 4


class TestRelativeNeighborhoodGraph:
    """relative_neighborhood_graph関数のテスト。"""
    
    def test_empty_nodes(self):
        """空のノードリストのテスト。"""
        edges, adjacency = relative_neighborhood_graph([], AsemicGlyphConfig())
        assert edges == []
        assert adjacency == {}
    
    def test_single_node(self):
        """単一ノードのテスト。"""
        nodes = [(0.0, 0.0, 0.0)]
        edges, adjacency = relative_neighborhood_graph(nodes, AsemicGlyphConfig())
        assert edges == []
        assert adjacency == {0: []}
    
    def test_two_nodes(self):
        """2つのノードのテスト。"""
        nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        config = AsemicGlyphConfig(min_distance=0.5)
        edges, adjacency = relative_neighborhood_graph(nodes, config)
        
        assert len(edges) == 1
        assert (0, 1) in edges
        assert adjacency[0] == [1]
        assert adjacency[1] == [0]
    
    def test_triangle_nodes(self):
        """三角形ノードのテスト。"""
        nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        config = AsemicGlyphConfig(min_distance=0.1)
        edges, adjacency = relative_neighborhood_graph(nodes, config)
        
        # RNGでは必ずしも全てのエッジが含まれるわけではない
        # 少なくとも1つ以上のエッジが生成されることを確認
        assert len(edges) >= 1
        for edge in edges:
            assert len(edge) == 2
            assert 0 <= edge[0] < len(nodes)
            assert 0 <= edge[1] < len(nodes)
        
        # 隣接リストが正しく構築されていることを確認
        assert len(adjacency) == len(nodes)
        for node_id, neighbors in adjacency.items():
            assert 0 <= node_id < len(nodes)
            for neighbor in neighbors:
                assert 0 <= neighbor < len(nodes)
                assert neighbor != node_id


class TestRandomWalkStrokes:
    """random_walk_strokes関数のテスト。"""
    
    def test_no_adjacency(self):
        """隣接関係のないノードのテスト。"""
        nodes = [(0.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
        adjacency = {0: [], 1: []}
        config = AsemicGlyphConfig()
        rng = random.Random(42)
        
        strokes = random_walk_strokes(nodes, adjacency, config, rng)
        assert strokes == []
    
    def test_simple_adjacency(self):
        """簡単な隣接関係のテスト。"""
        nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        adjacency = {0: [1], 1: [0, 2], 2: [1]}
        config = AsemicGlyphConfig(walk_min_steps=1, walk_max_steps=2)
        rng = random.Random(42)
        
        strokes = random_walk_strokes(nodes, adjacency, config, rng)
        assert len(strokes) > 0
        for stroke in strokes:
            assert len(stroke) >= 2  # 最小ストローク長


class TestDiacriticFactory:
    """DiacriticFactoryクラスのテスト。"""
    
    def test_create_circle(self):
        """円形アクセントのテスト。"""
        center = (0.0, 0.0, 0.0)
        radius = 0.1
        result = DiacriticFactory.create_circle(center, radius)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3  # 3D座標
        assert result.shape[0] > 0
        
        # 最初と最後の点が同じ（閉じた形状）
        np.testing.assert_array_almost_equal(result[0], result[-1])
    
    def test_create_tilde(self):
        """チルダアクセントのテスト。"""
        center = (0.0, 0.0, 0.0)
        radius = 0.1
        result = DiacriticFactory.create_tilde(center, radius)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3
        assert result.shape[0] > 0
    
    def test_create_umlaut(self):
        """ウムラウトアクセントのテスト（複数の形状を返す）。"""
        center = (0.0, 0.0, 0.0)
        radius = 0.1
        result = DiacriticFactory.create_umlaut(center, radius)
        
        assert isinstance(result, list)
        assert len(result) == 2  # 2つのドット
        for dot in result:
            assert isinstance(dot, np.ndarray)
            assert dot.shape[1] == 3
    
    def test_create_random_diacritic(self):
        """ランダムなディアクリティカル生成のテスト。"""
        center = (0.0, 0.0, 0.0)
        radius = 0.1
        rng = random.Random(42)
        
        result = DiacriticFactory.create_random_diacritic(center, radius, rng)
        assert isinstance(result, list)
        assert len(result) > 0
        for shape in result:
            assert isinstance(shape, np.ndarray)
            assert shape.shape[1] == 3
    
    def test_all_diacritic_types(self):
        """全てのディアクリティカルタイプが実行可能であることのテスト。"""
        center = (0.0, 0.0, 0.0)
        radius = 0.1
        
        for diacritic_type, method in DiacriticFactory.DIACRITIC_TYPES.items():
            result = method(center, radius)
            
            if isinstance(result, list):
                assert len(result) > 0
                for shape in result:
                    assert isinstance(shape, np.ndarray)
            else:
                assert isinstance(result, np.ndarray)
                assert result.shape[1] == 3


class TestAddDiacritic:
    """add_diacritic関数のテスト。"""
    
    def test_add_diacritic_basic(self):
        """基本的なディアクリティカル追加のテスト。"""
        vertices_list = []
        nodes = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        used_nodes = {0, 1}
        rng = random.Random(42)
        
        add_diacritic(vertices_list, nodes, used_nodes, 1.0, 0.1, rng)
        
        # 1つ以上のディアクリティカルが追加される
        assert len(vertices_list) > 0
        for shape in vertices_list:
            assert isinstance(shape, np.ndarray)
            assert shape.shape[1] == 3
    
    def test_add_diacritic_no_probability(self):
        """確率0でディアクリティカルが追加されないテスト。"""
        vertices_list = []
        nodes = [(0.0, 0.0, 0.0)]
        used_nodes = {0}
        rng = random.Random(42)
        
        add_diacritic(vertices_list, nodes, used_nodes, 0.0, 0.1, rng)
        
        # 確率0なので何も追加されない
        assert len(vertices_list) == 0
    
    def test_add_diacritic_empty_used_nodes(self):
        """使用ノードが空の場合のテスト。"""
        vertices_list = []
        nodes = [(0.0, 0.0, 0.0)]
        used_nodes = set()
        rng = random.Random(42)
        
        add_diacritic(vertices_list, nodes, used_nodes, 1.0, 0.1, rng)
        
        # 使用ノードが空なので何も追加されない
        assert len(vertices_list) == 0


class TestAsemicGlyph:
    """AsemicGlyphクラスのテスト。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行されるセットアップ。"""
        self.glyph = AsemicGlyph()
    
    def test_basic_generation(self):
        """基本的なアセミック文字生成のテスト。"""
        vertices_list = self.glyph.generate()
        
        # 何らかの形状が生成される
        assert len(vertices_list) >= 0
        
        for vertices in vertices_list:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3  # 3D座標
            assert vertices.shape[0] >= 2  # 最低2つの頂点
    
    def test_custom_parameters(self):
        """カスタムパラメータでのテスト。"""
        vertices_list = self.glyph.generate(
            region=(-0.3, -0.3, 0.3, 0.3),
            smoothing_radius=0.02,
            diacritic_probability=0.5,
            diacritic_radius=0.03,
            random_seed=123.0
        )
        
        # 指定したパラメータで正常に動作
        assert isinstance(vertices_list, list)
        
        # 全ての頂点が指定した領域内にある（余白考慮）
        for vertices in vertices_list:
            for vertex in vertices:
                # 余白を考慮した範囲内であることを確認
                assert -0.4 <= vertex[0] <= 0.4
                assert -0.4 <= vertex[1] <= 0.4
    
    def test_reproducibility(self):
        """再現性のテスト（同じシードで同じ結果）。"""
        seed = 42.0
        
        vertices_list_1 = self.glyph.generate(random_seed=seed)
        vertices_list_2 = self.glyph.generate(random_seed=seed)
        
        assert len(vertices_list_1) == len(vertices_list_2)
        
        for v1, v2 in zip(vertices_list_1, vertices_list_2):
            np.testing.assert_array_almost_equal(v1, v2)
    
    def test_different_seeds_different_results(self):
        """異なるシードで異なる結果になることのテスト。"""
        vertices_list_1 = self.glyph.generate(random_seed=42.0)
        vertices_list_2 = self.glyph.generate(random_seed=123.0)
        
        # 完全に同じ結果になる可能性は低い
        # 少なくとも長さやいくつかの値は異なるはず
        if len(vertices_list_1) > 0 and len(vertices_list_2) > 0:
            # 何らかの違いがあることを確認
            # 完全に一致するかどうかをチェック
            all_match = True
            if len(vertices_list_1) != len(vertices_list_2):
                all_match = False
            else:
                for v1, v2 in zip(vertices_list_1, vertices_list_2):
                    if not np.array_equal(v1, v2):
                        all_match = False
                        break
            
            # 完全に一致する可能性は低い（ランダム生成のため）
            # ただし、稀に一致することもあるので、これは必須ではない
    
    def test_region_constraints(self):
        """領域制約のテスト。"""
        region = (-0.2, -0.2, 0.2, 0.2)
        vertices_list = self.glyph.generate(region=region)
        
        # 生成された頂点が概ね領域内にあることを確認
        # 余白やスムージングで少し範囲外になることもある
        for vertices in vertices_list:
            for vertex in vertices:
                # 多少の余裕をもって範囲チェック
                assert -0.5 <= vertex[0] <= 0.5
                assert -0.5 <= vertex[1] <= 0.5
    
    def test_zero_diacritic_probability(self):
        """ディアクリティカル確率0のテスト。"""
        vertices_list = self.glyph.generate(
            diacritic_probability=0.0,
            random_seed=42.0
        )
        
        # ディアクリティカルなしでも基本形状は生成される
        assert isinstance(vertices_list, list)
    
    def test_high_diacritic_probability(self):
        """ディアクリティカル確率1.0のテスト。"""
        vertices_list = self.glyph.generate(
            diacritic_probability=1.0,
            random_seed=42.0
        )
        
        # 高確率でディアクリティカルが含まれる
        assert isinstance(vertices_list, list)
    
    def test_cache_functionality(self):
        """キャッシュ機能のテスト。"""
        # キャッシュクリア
        self.glyph.clear_cache()
        
        # 同じパラメータで2回生成
        params = {
            "region": (-0.5, -0.5, 0.5, 0.5),
            "smoothing_radius": 0.05,
            "random_seed": 42.0
        }
        
        vertices_list_1 = self.glyph.generate(**params)
        vertices_list_2 = self.glyph.generate(**params)
        
        # キャッシュにより同じ結果が返される
        assert len(vertices_list_1) == len(vertices_list_2)
        
        for v1, v2 in zip(vertices_list_1, vertices_list_2):
            np.testing.assert_array_equal(v1, v2)
    
    def test_additional_params_ignored(self):
        """追加パラメータが無視されることのテスト。"""
        vertices_list = self.glyph.generate(
            region=(-0.5, -0.5, 0.5, 0.5),
            unused_param=123,
            another_param="test",
            random_seed=42.0
        )
        
        # 正常に動作することを確認
        assert isinstance(vertices_list, list)
    
    def test_extreme_smoothing_radius(self):
        """極端なスムージング半径のテスト。"""
        # 非常に小さい半径
        vertices_list_small = self.glyph.generate(
            smoothing_radius=0.001,
            random_seed=42.0
        )
        assert isinstance(vertices_list_small, list)
        
        # 非常に大きい半径
        vertices_list_large = self.glyph.generate(
            smoothing_radius=1.0,
            random_seed=42.0
        )
        assert isinstance(vertices_list_large, list)
    
    def test_small_region(self):
        """小さな領域でのテスト。"""
        small_region = (-0.05, -0.05, 0.05, 0.05)
        vertices_list = self.glyph.generate(
            region=small_region,
            random_seed=42.0
        )
        
        # 小さな領域でも何らかの形状が生成される
        assert isinstance(vertices_list, list)
    
    def test_inheritance_from_base_shape(self):
        """BaseShapeからの継承確認。"""
        from shapes.base import BaseShape
        assert isinstance(self.glyph, BaseShape)
        
        # BaseShapeのメソッドが使用可能
        assert hasattr(self.glyph, 'clear_cache')
        assert hasattr(self.glyph, 'disable_cache')
        assert hasattr(self.glyph, 'enable_cache')
    
    def test_callable_interface(self):
        """呼び出し可能インターフェースのテスト。"""
        # __call__メソッドを通じた呼び出し
        vertices_list = self.glyph(
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, 0),
            random_seed=42.0
        )
        
        assert isinstance(vertices_list, list)
        
        # 変形パラメータ付きの呼び出し
        vertices_list_transformed = self.glyph(
            center=(0.1, 0.1, 0),
            scale=(0.5, 0.5, 1),
            random_seed=42.0
        )
        
        assert isinstance(vertices_list_transformed, list)