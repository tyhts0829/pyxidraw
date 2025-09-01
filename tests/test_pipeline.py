"""
effects.pipeline モジュールのテスト
"""
import numpy as np
import pytest

from effects.pipeline import EffectPipeline
from effects.rotation import Rotation
from effects.scaling import Scaling
from effects.translation import Translation
from effects.subdivision import Subdivision
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestEffectPipeline:
    def test_init_empty(self):
        """空パイプライン初期化テスト"""
        pipeline = EffectPipeline()
        assert pipeline is not None
        assert len(pipeline.effects) == 0

    def test_init_with_effects(self):
        """エフェクト付きパイプライン初期化テスト"""
        effects = [Rotation(), Scaling(), Translation()]
        pipeline = EffectPipeline(effects)
        assert pipeline is not None
        assert len(pipeline.effects) == 3

    def test_single_effect_pipeline(self, simple_geometry):
        """単一エフェクトパイプラインテスト"""
        pipeline = EffectPipeline([Rotation()])
        result = pipeline(simple_geometry, rotate=(0.0, 0.0, 0.25))
        assert isinstance(result, Geometry)

    def test_two_effect_pipeline(self, simple_geometry):
        """二重エフェクトパイプラインテスト"""
        pipeline = EffectPipeline([Scaling(), Translation()])
        result = pipeline(
            simple_geometry,
            scale=(2.0, 2.0, 2.0),
            offset_x=5.0,
            offset_y=3.0,
            offset_z=1.0
        )
        assert isinstance(result, Geometry)

    def test_multiple_effect_pipeline(self, simple_geometry):
        """複数エフェクトパイプラインテスト"""
        pipeline = EffectPipeline([
            Rotation(),
            Scaling(),
            Translation(),
            Subdivision()
        ])
        result = pipeline(
            simple_geometry,
            rotate=(0.0, 0.0, 0.1),
            scale=(1.5, 1.5, 1.5),
            offset_x=2.0,
            offset_y=1.0,
            offset_z=0.5,
            n_divisions=0.3
        )
        assert isinstance(result, Geometry)

    def test_pipeline_order_matters(self, simple_geometry):
        """パイプライン順序重要性テスト"""
        # スケール → 移動
        pipeline1 = EffectPipeline([Scaling(), Translation()])
        result1 = pipeline1(
            simple_geometry,
            scale=(2.0, 2.0, 2.0),
            offset_x=5.0
        )
        
        # 移動 → スケール
        pipeline2 = EffectPipeline([Translation(), Scaling()])
        result2 = pipeline2(
            simple_geometry,
            offset_x=5.0,
            scale=(2.0, 2.0, 2.0)
        )
        
        assert isinstance(result1, Geometry)
        assert isinstance(result2, Geometry)
        # 順序が異なるので結果も異なる
        assert not np.allclose(result1.coords, result2.coords)

    def test_add_effect(self):
        """エフェクト追加テスト"""
        pipeline = EffectPipeline()
        assert len(pipeline.effects) == 0
        
        pipeline.add(Rotation())
        assert len(pipeline.effects) == 1
        
        pipeline.add(Scaling())
        assert len(pipeline.effects) == 2

    def test_empty_pipeline_passthrough(self, simple_geometry):
        """空パイプライン通過テスト"""
        pipeline = EffectPipeline()
        result = pipeline(simple_geometry)
        assert isinstance(result, Geometry)
        # 空パイプラインなので元と同じ
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_complex_pipeline(self, simple_geometry):
        """複雑なパイプラインテスト"""
        pipeline = EffectPipeline([
            Subdivision(),  # 細分化
            Rotation(),     # 回転
            Scaling(),      # スケール
            Translation()   # 移動
        ])
        
        result = pipeline(
            simple_geometry,
            n_divisions=0.5,
            rotate=(0.1, 0.2, 0.3),
            scale=(1.5, 0.8, 2.0),
            offset_x=10.0,
            offset_y=-5.0,
            offset_z=3.0
        )
        assert isinstance(result, Geometry)

    def test_duplicate_effect_types(self, simple_geometry):
        """重複エフェクトタイプテスト"""
        # 同じタイプのエフェクトを複数含むパイプライン
        pipeline = EffectPipeline([
            Translation(),
            Rotation(),
            Translation()  # 再度移動
        ])
        
        result = pipeline(
            simple_geometry,
            rotate=(0.0, 0.0, 0.25),
            offset_x=2.0,
            offset_y=3.0
        )
        assert isinstance(result, Geometry)

    def test_parameter_propagation(self, simple_geometry):
        """パラメータ伝播テスト"""
        pipeline = EffectPipeline([Rotation(), Scaling()])
        
        # 各エフェクトに必要なパラメータを渡す
        result = pipeline(
            simple_geometry,
            rotate=(0.0, 0.0, 0.25),    # Rotation用
            scale=(2.0, 2.0, 2.0),      # Scaling用
            center=(0.5, 0.5, 0.0)      # 両方で使用される可能性
        )
        assert isinstance(result, Geometry)

    def test_pipeline_with_defaults(self, simple_geometry):
        """デフォルトパラメータパイプラインテスト"""
        pipeline = EffectPipeline([Rotation(), Scaling(), Translation()])
        
        # 一部のパラメータのみ指定
        result = pipeline(simple_geometry, scale=(2.0, 2.0, 2.0))
        assert isinstance(result, Geometry)

    def test_progressive_transformation(self, simple_geometry):
        """段階的変換テスト"""
        # 各段階での結果を確認
        rotation = Rotation()
        scaling = Scaling()
        translation = Translation()
        
        # 段階1: 回転のみ
        step1 = rotation(simple_geometry, rotate=(0.0, 0.0, 0.25))
        
        # 段階2: 回転 + スケール
        step2 = scaling(step1, scale=(2.0, 2.0, 2.0))
        
        # 段階3: 回転 + スケール + 移動
        step3 = translation(step2, offset_x=5.0, offset_y=3.0)
        
        # パイプラインで一度に実行
        pipeline = EffectPipeline([rotation, scaling, translation])
        pipeline_result = pipeline(
            simple_geometry,
            rotate=(0.0, 0.0, 0.25),
            scale=(2.0, 2.0, 2.0),
            offset_x=5.0,
            offset_y=3.0
        )
        
        # 結果が同じになることを確認
        np.testing.assert_allclose(step3.coords, pipeline_result.coords, rtol=1e-6)