"""
どこで: `engine.ui.hud` パッケージ。
何を: HUD 表示の設定と項目定義（フィールド名）を提供する。
なぜ: HUD の有効/無効や表示項目の選択を宣言的に制御し、将来の拡張に備えるため。
"""

from __future__ import annotations

from .config import HUDConfig
from .fields import CACHE_EFFECT, CACHE_SHAPE, CPU, FPS, LINE, MEM, VERTEX

__all__ = [
    "HUDConfig",
    "FPS",
    "VERTEX",
    "LINE",
    "CPU",
    "MEM",
    "CACHE_SHAPE",
    "CACHE_EFFECT",
]
