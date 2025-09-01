"""
api.effect_chain (E) モジュールのテスト

新しいエフェクトチェーンAPIの基本的な動作を検証する。
効率性と保守性を重視した最小限のテスト実装。
"""
import math

import numpy as np
import pytest

from api import E, G
from api.geometry_api import GeometryAPI


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometryAPIオブジェクト"""
    return G.polygon(n_sides=3)  # 三角形


@pytest.fixture
def complex_geometry():
    """テスト用の複雑なGeometryAPIオブジェクト"""
    return G.sphere(subdivisions=0.3)


class TestEffectChain:
    """エフェクトチェーン（E）のテストクラス"""

    def test_basic_chain(self, simple_geometry):
        """基本的なエフェクトチェーンのテスト"""
        result = (E.add(simple_geometry)
                  .rotation(rotate=(0, 0, 0.1))
                  .scaling(scale=(1.5, 1.5, 1.5))
                  .result())
        
        assert isinstance(result, GeometryAPI)
        assert result is not None

    def test_rotation_effect(self, simple_geometry):
        """回転エフェクトのテスト"""
        result = E.add(simple_geometry).rotation(
            center=(0, 0, 0),
            rotate=(0, 0, math.pi/4)
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_scaling_effect(self, simple_geometry):
        """スケーリングエフェクトのテスト"""
        result = E.add(simple_geometry).scaling(
            center=(0, 0, 0),
            scale=(2, 2, 2)
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_translation_effect(self, simple_geometry):
        """平行移動エフェクトのテスト"""
        result = E.add(simple_geometry).translation(
            offset_x=10,
            offset_y=20,
            offset_z=0
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_subdivision_effect(self, simple_geometry):
        """細分化エフェクトのテスト"""
        result = E.add(simple_geometry).subdivision(
            n_divisions=0.5
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_noise_effect(self, simple_geometry):
        """ノイズエフェクトのテスト"""
        result = E.add(simple_geometry).noise(
            intensity=0.3,
            frequency=1.0,
            time=0.0
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_buffer_effect(self, simple_geometry):
        """バッファエフェクトのテスト"""
        result = E.add(simple_geometry).buffer(
            distance=0.1,
            join_style=0.5,
            resolution=0.5
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_filling_effect(self, simple_geometry):
        """塗りつぶしエフェクトのテスト"""
        result = E.add(simple_geometry).filling(
            pattern="lines",
            density=0.5,
            angle=0.0
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_transform_effect(self, simple_geometry):
        """変換エフェクトのテスト"""
        result = E.add(simple_geometry).transform(
            center=(0, 0, 0),
            scale=(1.5, 1.5, 1.5),
            rotate=(0, 0, 0.1)
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_array_effect(self, simple_geometry):
        """配列エフェクトのテスト"""
        result = E.add(simple_geometry).array(
            n_duplicates=0.3,
            offset=(5, 5, 0),
            rotate=(0, 0, 0.1),
            scale=(0.9, 0.9, 0.9)
        ).result()
        
        assert isinstance(result, GeometryAPI)

    def test_complex_chain(self, complex_geometry):
        """複雑なエフェクトチェーンのテスト"""
        result = (E.add(complex_geometry)
                  .rotation(rotate=(0.1, 0.1, 0.1))
                  .scaling(scale=(1.2, 1.2, 1.2))
                  .subdivision(n_divisions=0.6)
                  .noise(intensity=0.2, time=0.5)
                  .buffer(distance=0.05)
                  .result())
        
        assert isinstance(result, GeometryAPI)

    def test_empty_chain(self, simple_geometry):
        """空のチェーン（E.add().result()のみ）のテスト"""
        result = E.add(simple_geometry).result()
        
        assert isinstance(result, GeometryAPI)

    def test_single_effect_chains(self, simple_geometry):
        """単一エフェクトチェーンのテスト"""
        # 各エフェクトを個別にテスト
        effects_to_test = [
            lambda g: E.add(g).rotation(rotate=(0, 0, 0.1)).result(),
            lambda g: E.add(g).scaling(scale=(1.1, 1.1, 1.1)).result(),
            lambda g: E.add(g).translation(offset_x=1).result(),
            lambda g: E.add(g).subdivision(n_divisions=0.3).result(),
            lambda g: E.add(g).noise(intensity=0.1).result(),
        ]
        
        for effect_func in effects_to_test:
            result = effect_func(simple_geometry)
            assert isinstance(result, GeometryAPI)


class TestEffectChainPerformance:
    """エフェクトチェーンのパフォーマンステスト"""

    def test_chain_caching(self, simple_geometry):
        """チェーンキャッシュのテスト（同じパラメータでの高速化）"""
        # 同じパラメータで複数回実行
        chain_func = lambda: (E.add(simple_geometry)
                              .rotation(rotate=(0, 0, 0.1))
                              .scaling(scale=(1.5, 1.5, 1.5))
                              .result())
        
        result1 = chain_func()
        result2 = chain_func()
        
        # 両方とも正常に実行されることを確認
        assert isinstance(result1, GeometryAPI)
        assert isinstance(result2, GeometryAPI)

    def test_different_parameters_no_cache(self, simple_geometry):
        """異なるパラメータでのキャッシュなし動作確認"""
        result1 = E.add(simple_geometry).rotation(rotate=(0, 0, 0.1)).result()
        result2 = E.add(simple_geometry).rotation(rotate=(0, 0, 0.2)).result()
        
        assert isinstance(result1, GeometryAPI)
        assert isinstance(result2, GeometryAPI)