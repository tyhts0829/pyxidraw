from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseEffect


class EffectPipeline:
    """複数のエフェクトを連結するためのパイプラインです。"""
    
    def __init__(self, effects: Sequence[BaseEffect] | None = None):
        """初期エフェクトでパイプラインを初期化します。
        
        Args:
            effects: 順番に適用するエフェクトのシーケンス
        """
        self.effects = list(effects) if effects else []
    
    def add(self, effect: BaseEffect) -> EffectPipeline:
        """パイプラインにエフェクトを追加します。
        
        Args:
            effect: 追加するエフェクト
            
        Returns:
            メソッドチェーンのための自身
        """
        self.effects.append(effect)
        return self
    
    def remove(self, effect: BaseEffect) -> EffectPipeline:
        """パイプラインからエフェクトを除去します。
        
        Args:
            effect: 除去するエフェクト
            
        Returns:
            メソッドチェーンのための自身
        """
        if effect in self.effects:
            self.effects.remove(effect)
        return self
    
    def clear(self) -> EffectPipeline:
        """パイプラインからすべてのエフェクトをクリアします。
        
        Returns:
            メソッドチェーンのための自身
        """
        self.effects.clear()
        return self
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """パイプライン内のすべてのエフェクトを順次適用します。
        
        Args:
            vertices_list: 入力頂点配列
            **params: すべてのエフェクトに渡されるパラメータ
            
        Returns:
            変換された頂点配列
        """
        result = vertices_list
        for effect in self.effects:
            result = effect(result, **params)
        return result
    
    def __call__(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """パイプラインを適用します（applyメソッドのエイリアス）。"""
        return self.apply(vertices_list, **params)
    
    def __len__(self) -> int:
        """パイプライン内のエフェクト数を取得します。"""
        return len(self.effects)
    
    def __getitem__(self, index: int) -> BaseEffect:
        """インデックスのエフェクトを取得します。"""
        return self.effects[index]
    
    def __iter__(self):
        """エフェクトを反復します。"""
        return iter(self.effects)
    
    def clear_all_caches(self):
        """パイプライン内のすべてのエフェクトのキャッシュをクリアします。"""
        for effect in self.effects:
            effect.clear_cache()