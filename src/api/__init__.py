"""
どこで: `api` 入口（高レベル公開 API）。
何を: 形状 `G`・エフェクト `E`・装飾子 `shape/effect`・`Geometry` などを再輸出。
なぜ: 利用者が単一名前空間から形状生成→パイプライン適用→実行まで完結できるようにするため。

PyxiDraw 次期API（関数・パイプライン・Geometry統一）

Usage:
    from api import G, E, Geometry

    g = G.sphere(subdivisions=0.5)
    g = g.scale(100, 100, 100).translate(100, 100, 0)

    pipeline = (E.pipeline
                  .displace(amplitude_mm=0.3)
                  .fill(density=0.5)
                  .build())
    result = pipeline(g)
"""

from effects.registry import effect as effect  # 公開唯一経路（api.effect）

# コアクラス（高度な使用）
from engine.core.geometry import Geometry
from shapes.registry import (
    shape as shape,
)  # 公開唯一経路（api.shape）。shape_registry では再輸出しない。

from .effects import E

# 主要API
from .shapes import G, ShapesAPI
from .sketch import run_sketch as run
from .sketch import run_sketch as run_sketch

__all__ = [
    # メインAPI
    "G",  # 形状ファクトリ
    "E",  # エフェクトファクトリ
    "shape",  # ユーザー拡張用デコレータ（唯一の公開経路）
    "effect",  # ユーザー拡張用デコレータ（唯一の公開経路）
    "run_sketch",  # 実行（詳細指定）
    "run",  # 実行（エイリアス、簡易）
    # クラス（高度な使用）
    "ShapesAPI",
    "Geometry",
    # シリアライズ補助（削除済み：外部保存/復元/検証は提供しない）
]

# バージョン情報
__version__ = "2025.09"
__api_version__ = "6.0"
