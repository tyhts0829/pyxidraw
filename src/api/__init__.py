"""
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

from .effects import E, from_spec, to_spec, validate_spec

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
    # シリアライズ補助
    "to_spec",
    "from_spec",
    "validate_spec",
]

# バージョン情報
__version__ = "2025.09"
__api_version__ = "6.0"
