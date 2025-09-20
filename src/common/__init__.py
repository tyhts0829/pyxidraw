"""
どこで: `common` パッケージ。
何を: shapes/effects 双方で使う軽量ユーティリティ（BaseRegistry など）。
なぜ: API 層から再利用する共通基盤を分離し、依存の向きを単純化するため。
"""

from .base_registry import BaseRegistry

__all__ = [
    "BaseRegistry",
]
