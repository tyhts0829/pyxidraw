"""
形状レジストリシステム - レジストリパターンによる拡張可能な形状管理。
統一されたレジストリシステムを使用するための互換性レイヤー。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type, Union

from engine.core.geometry import Geometry
from shapes.registry import shape, get_shape, list_shapes, is_shape_registered, clear_registry
from shapes import registry as _shapes_registry


register_shape = shape


def get_shape_generator(name: str):
    """登録された形状生成器を取得。（互換性のための関数）"""
    try:
        return get_shape(name)
    except KeyError:
        raise ValueError(f"未登録のシェイプ名です: {name}")


def list_registered_shapes() -> List[str]:
    """登録されているすべての形状名を返す。（互換性のための関数）"""
    return list_shapes()


def unregister_shape(name: str):
    """形状の登録を解除（主にテスト用）。"""
    try:
        _shapes_registry.get_registry().unregister(name)
    except Exception:
        # 存在しない場合は無視（安全側）
        return


class CustomShape(ABC):
    """ユーザー定義形状の基底クラス。"""
    
    @abstractmethod
    def generate(self, **params) -> Geometry:
        """形状を生成する抽象メソッド。
        
        Args:
            **params: 形状生成パラメータ
            
        Returns:
            Geometry: 生成された形状データ
        """
        pass


class ValidatedCustomShape(CustomShape):
    """パラメータ検証機能付きカスタム形状基底クラス。"""
    
    def get_default_params(self) -> dict:
        """デフォルトパラメータを定義。"""
        return {}
    
    def validate_params(self, **params) -> dict:
        """パラメータを検証し、デフォルト値を適用。"""
        defaults = self.get_default_params()
        validated = {**defaults, **params}
        self._validate_impl(validated)
        return validated
    
    def _validate_impl(self, params: dict):
        """パラメータ検証ロジック（サブクラスでオーバーライド可能）。"""
        pass
    
    def generate(self, **params) -> Geometry:
        validated_params = self.validate_params(**params)
        return self._generate_impl(**validated_params)
    
    @abstractmethod
    def _generate_impl(self, **params) -> Geometry:
        """実際の生成ロジック（サブクラスで実装）。"""
        pass
