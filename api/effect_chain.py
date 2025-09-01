"""
EffectChain - エフェクトチェーン実装（次期API仕様）
E.add()から始まるメソッドチェーンAPIと高性能キャッシュを提供。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, List
from weakref import WeakValueDictionary
import effects  # 登録を副作用で実行
from effects.registry import get_effect as get_registered_effect, list_effects as list_registered_effects
from common.base_registry import BaseRegistry

from .geometry_api import GeometryAPI


class EffectStep:
    """エフェクトチェーンの1つのステップを表すクラス。"""

    def __init__(self, effect_name: str, params: dict):
        self.effect_name = effect_name
        self.params = params
        self.params_hash = self._compute_hash(params)

    def _compute_hash(self, params: dict) -> int:
        """パラメータのハッシュ値を計算。"""

        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, (list, tuple)):
                return tuple(make_hashable(item) for item in obj)
            else:
                return obj

        hashable_params = make_hashable(params)
        return hash(hashable_params)

    def __eq__(self, other):
        if not isinstance(other, EffectStep):
            return False
        return self.effect_name == other.effect_name and self.params_hash == other.params_hash

    def __hash__(self):
        return hash((self.effect_name, self.params_hash))


class EffectChain:
    """エフェクトチェーン実装クラス。"""

    # カスタムエフェクトのレジストリ
    _custom_effects: Dict[str, Callable] = {}

    # 統合キャッシュシステム
    _chain_cache: WeakValueDictionary[tuple, GeometryAPI] = WeakValueDictionary()  # チェーン全体の結果
    _step_cache: Dict[tuple, GeometryAPI] = {}  # 個別ステップの結果
    _step_cache_maxsize: int = 1024

    def __init__(self, base_geometry: GeometryAPI):
        """
        Args:
            base_geometry: ベースとなるGeometryAPI
        """
        self._base_geometry = base_geometry
        self._steps: List[EffectStep] = []

    def _get_cache_key(self) -> tuple:
        """キャッシュキーを生成。"""
        base_guid = self._base_geometry.guid
        steps_hash = tuple(self._steps)
        return (base_guid, steps_hash)

    def _apply_effects(self) -> GeometryAPI:
        """全エフェクトを順次適用（統一API処理）。"""
        current_api = self._base_geometry
        
        for step in self._steps:
            current_api = self._apply_single_effect(current_api, step)
        
        return current_api
    
    def _apply_single_effect(self, geometry_api: GeometryAPI, step: EffectStep) -> GeometryAPI:
        """単一エフェクトの適用（統合キャッシュ付き）。"""
        # ステップレベルのキャッシュキー
        step_key = (geometry_api.guid, step.effect_name, step.params_hash)
        
        # ステップキャッシュをチェック
        if step_key in self._step_cache:
            return self._step_cache[step_key]
        
        # キャッシュサイズ管理（LRU風に最古のエントリを削除）
        if len(self._step_cache) >= self._step_cache_maxsize:
            # 最初のエントリを削除（簡易LRU）
            oldest_key = next(iter(self._step_cache))
            del self._step_cache[oldest_key]
        
        # エフェクト適用
        if step.effect_name in self._custom_effects:
            result = self._custom_effects[step.effect_name](geometry_api, **step.params)
        else:
            # 標準エフェクトとしてレジストリから解決
            result = self._apply_standard_effect(geometry_api, step)
        
        # 結果をキャッシュ
        self._step_cache[step_key] = result
        return result
    
    def _apply_standard_effect(self, geometry_api: GeometryAPI, step: EffectStep) -> GeometryAPI:
        """標準エフェクトの適用（ピュア処理）。"""
        # GeometryAPI → numpy配列変換
        coords, offsets = geometry_api.data.as_arrays()
        
        # 純粋なエフェクト処理（キャッシュなし）
        effect_class = get_registered_effect(step.effect_name)
        effect_instance = effect_class()  # disable_cache()呼び出し不要
        new_coords, new_offsets = effect_instance.apply(coords, offsets, **step.params)
        
        # numpy配列 → GeometryAPI変換
        from engine.core.geometry_data import GeometryData
        return GeometryAPI(GeometryData(new_coords, new_offsets))
    
    

    def _get_result(self) -> GeometryAPI:
        """結果を取得（統合キャッシュ使用）。"""
        cache_key = self._get_cache_key()

        if cache_key in self._chain_cache:
            return self._chain_cache[cache_key]

        # エフェクトを適用して結果を計算
        result = self._apply_effects()
        self._chain_cache[cache_key] = result
        return result

    def _add_step(self, effect_name: str, params: dict | None = None, **kwargs) -> "EffectChain":
        """新しいエフェクトステップを追加。"""
        if params is None:
            params = kwargs if kwargs else {}
        else:
            params = {**params, **kwargs}
            
        new_chain = EffectChain(self._base_geometry)
        new_chain._steps = self._steps.copy()
        new_chain._steps.append(EffectStep(effect_name, params))
        return new_chain

    # === 標準エフェクトメソッド ===

    def noise(
        self,
        intensity: float = 0.5,
        frequency: tuple[float, float, float] | float = (0.5, 0.5, 0.5),
        time: float = 0.0,
        **params,
    ) -> "EffectChain":
        """ノイズエフェクトを追加。"""
        all_params = {"intensity": intensity, "frequency": frequency, "time": time, **params}
        return self._add_step("noise", all_params)

    def filling(self, pattern: str = "lines", density: float = 0.5, angle: float = 0.0, **params) -> "EffectChain":
        """塗りつぶしエフェクトを追加。"""
        all_params = {"pattern": pattern, "density": density, "angle": angle, **params}
        return self._add_step("filling", all_params)

    def rotation(
        self, center: tuple[float, float, float] = (0, 0, 0), rotate: tuple[float, float, float] = (0, 0, 0), **params
    ) -> "EffectChain":
        """回転エフェクトを追加。"""
        all_params = {"center": center, "rotate": rotate, **params}
        return self._add_step("rotation", all_params)

    def scaling(
        self, center: tuple[float, float, float] = (0, 0, 0), scale: tuple[float, float, float] = (1, 1, 1), **params
    ) -> "EffectChain":
        """拡大縮小エフェクトを追加。"""
        all_params = {"center": center, "scale": scale, **params}
        return self._add_step("scaling", all_params)

    def translation(
        self, offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0, **params
    ) -> "EffectChain":
        """平行移動エフェクトを追加。"""
        all_params = {"offset_x": offset_x, "offset_y": offset_y, "offset_z": offset_z, **params}
        return self._add_step("translation", all_params)

    def transform(self, **params) -> "EffectChain":
        """複合変換エフェクトを追加。"""
        return self._add_step("transform", params)

    def subdivision(self, n_divisions: float = 0.5, **params) -> "EffectChain":
        """細分化エフェクトを追加。"""
        all_params = {"n_divisions": n_divisions, **params}
        return self._add_step("subdivision", all_params)

    def extrude(
        self,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        distance: float = 0.5,
        scale: float = 0.5,
        subdivisions: float = 0.5,
        **params,
    ) -> "EffectChain":
        """押し出しエフェクトを追加。"""
        all_params = {
            "direction": direction,
            "distance": distance,
            "scale": scale,
            "subdivisions": subdivisions,
            **params,
        }
        return self._add_step("extrude", all_params)

    def buffer(
        self, distance: float = 0.5, join_style: float = 0.5, resolution: float = 0.5, **params
    ) -> "EffectChain":
        """バッファエフェクトを追加。"""
        all_params = {"distance": distance, "join_style": join_style, "resolution": resolution, **params}
        return self._add_step("buffer", all_params)

    def array(
        self,
        n_duplicates: float = 0.5,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float] = (0.5, 0.5, 0.5),
        scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        **params,
    ) -> "EffectChain":
        """配列エフェクトを追加。"""
        all_params = {
            "n_duplicates": n_duplicates,
            "offset": offset,
            "rotate": rotate,
            "scale": scale,
            "center": center,
            **params,
        }
        return self._add_step("array", all_params)

    # === 拡張機能 ===

    def apply(self, func: Callable[[GeometryAPI], GeometryAPI]) -> "EffectChain":
        """ワンショット効果を適用。

        Args:
            func: GeometryAPIを受け取りGeometryAPIを返す関数

        Returns:
            新しいEffectChain
        """
        func_id = f"apply_{id(func)}"
        self._custom_effects[func_id] = func
        return self._add_step(func_id, {})

    # === 動的メソッド生成 ===

    def __getattr__(self, name: str):
        """未定義メソッドの動的生成。"""
        if name in self._custom_effects:

            def custom_effect(**params):
                return self._add_step(name, params)

            return custom_effect
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # === 結果取得 ===

    def result(self) -> GeometryAPI:
        """最終結果を取得。"""
        return self._get_result()

    def __call__(self) -> GeometryAPI:
        """() で結果を取得。"""
        return self.result()

    # 仕様書との互換性のため、チェーン終端で自動的にGeometryAPIを返す
    def __repr__(self) -> str:
        """文字列表現。"""
        steps_str = " → ".join(step.effect_name for step in self._steps)
        return f"EffectChain({steps_str})"

    # プロパティとして結果を取得（仕様書の例に対応）
    @property
    def geometry(self) -> GeometryAPI:
        """プロパティとして結果を取得。"""
        return self._get_result()

    def steps(self) -> List[str]:
        """適用されるエフェクトのリストを取得。"""
        return [step.effect_name for step in self._steps]


class EffectFactory:
    """エフェクトファクトリクラス（E）。"""

    # カスタムエフェクトのグローバルレジストリ
    _global_custom_effects: Dict[str, Callable] = {}

    @staticmethod
    def add(geometry: GeometryAPI) -> EffectChain:
        """エフェクトチェーンを開始。

        Args:
            geometry: ベースとなるGeometryAPI

        Returns:
            EffectChain: 新しいエフェクトチェーン
        """
        return EffectChain(geometry)

    @classmethod
    def register(cls, name: str | None = None):
        """カスタムエフェクト登録デコレータ。

        name を省略した場合は関数名をスネークケース化して使用。
        """

        def decorator(func: Callable[[GeometryAPI], GeometryAPI]):
            effect_name = BaseRegistry._normalize_key(name) if name else BaseRegistry._normalize_key(func.__name__)
            cls._global_custom_effects[effect_name] = func
            EffectChain._custom_effects[effect_name] = func
            return func

        return decorator

    @classmethod
    def list_effects(cls) -> Dict[str, List[str]]:
        """利用可能なエフェクトのリストを取得。"""
        return {"standard": list_registered_effects(), "custom": list(cls._global_custom_effects.keys())}

    @property
    def pipeline(self):
        """新しいパイプラインビルダーを作成"""
        from .effect_pipeline import PipelineBuilder
        return PipelineBuilder()
    
    @classmethod
    def create_pipeline(cls):
        """パイプラインビルダーの明示的な作成"""
        from .effect_pipeline import PipelineBuilder
        return PipelineBuilder()

    @classmethod
    def clear_cache(cls):
        """統合キャッシュをクリア。"""
        EffectChain._chain_cache.clear()
        EffectChain._step_cache.clear()
    
    @classmethod
    def cache_info(cls) -> dict:
        """キャッシュ統計情報を取得。"""
        return {
            'chain_cache_size': len(EffectChain._chain_cache),
            'step_cache_size': len(EffectChain._step_cache),
            'step_cache_maxsize': EffectChain._step_cache_maxsize
        }


# シングルトンインスタンス（E）
E = EffectFactory()
