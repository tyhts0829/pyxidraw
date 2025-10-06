"""
どこで: `engine.ui.hud.fields`。
何を: HUD 表示に用いる標準フィールド名（ラベルキー）を定義する。
なぜ: 項目名の重複や表記ゆれを避け、順序指定や参照を安定化するため。
"""

from __future__ import annotations

# 表示キー（ラベルの左側に出るキー文字列）
FPS = "FPS"
VERTEX = "VERTEX"
LINE = "LINE"
CPU = "CPU"
MEM = "MEM"
CACHE_SHAPE = "CACHE/SHAPE"
CACHE_EFFECT = "CACHE/EFFECT"

__all__ = [
    "FPS",
    "VERTEX",
    "LINE",
    "CPU",
    "MEM",
    "CACHE_SHAPE",
    "CACHE_EFFECT",
]
