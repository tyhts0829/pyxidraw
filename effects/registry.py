"""
エフェクトレジストリ - 統一されたレジストリシステム
shapes/ と対称性を保った @effect デコレータの実装。

改善点:
- 名前省略可: `@effect` や `@effect()` でクラス名から自動推論（例: `Noise` -> `noise`）。
- エイリアス対応: `@effect("new", aliases=["old"])` で移行しやすく。
"""

from typing import Dict, Type, Callable, Any
from .base import BaseEffect
from common.base_registry import BaseRegistry


# 統一されたレジストリシステム
_effect_registry = BaseRegistry()


def effect(arg: str | Type[BaseEffect] | None = None):  # type: ignore[name-defined]
    """エフェクトをレジストリに登録するデコレータ。

    使い方:
    - 省略名で登録: `@effect` または `@effect()` -> クラス名から推論（`Noise` -> `noise`）
    - 明示名で登録: `@effect("noise")` も可能（必要な場合のみ）
    """
    if callable(arg):
        return _effect_registry.register(None)(arg)  # type: ignore[arg-type]
    name = arg if isinstance(arg, str) else None
    return _effect_registry.register(name)


def get_effect(name: str) -> Type[BaseEffect]:
    """登録されたエフェクトクラスを取得。
    
    Args:
        name: エフェクト名
        
    Returns:
        エフェクトクラス
        
    Raises:
        KeyError: エフェクトが登録されていない場合
    """
    return _effect_registry.get(name)


def list_effects() -> list[str]:
    """登録されているエフェクトの一覧を取得。
    
    Returns:
        エフェクト名のリスト
    """
    return _effect_registry.list_all()


def is_effect_registered(name: str) -> bool:
    """エフェクトが登録されているかチェック。
    
    Args:
        name: エフェクト名
        
    Returns:
        登録されている場合True
    """
    return _effect_registry.is_registered(name)


def clear_registry():
    """レジストリをクリア（テスト用）。"""
    _effect_registry.clear()


def get_registry() -> BaseRegistry:
    """レジストリインスタンスを取得（ファクトリクラス用）。
    
    Returns:
        レジストリインスタンス
    """
    return _effect_registry
