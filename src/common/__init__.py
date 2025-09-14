"""
共通基盤モジュール
shapes/ と effects/ の両方で使用するレジストリ等の軽量ユーティリティ。
"""

from .base_registry import BaseRegistry, CacheableRegistry

__all__ = [
    "BaseRegistry",
    "CacheableRegistry",
]
