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

# 主要API
from .shape_factory import G, ShapeFactory
from .pipeline import E, to_spec, from_spec, validate_spec
from .runner import run_sketch as run_sketch, run_sketch as run

# コアクラス（高度な使用）
from engine.core.geometry import Geometry

__all__ = [
    # メインAPI
    "G",           # 形状ファクトリ
    "E",           # エフェクトファクトリ
    "run_sketch",  # 実行（詳細指定）
    "run",         # 実行（エイリアス、簡易）
    
    # クラス（高度な使用）
    "ShapeFactory",
    "Geometry",
    # シリアライズ補助
    "to_spec",
    "from_spec",
    "validate_spec",
]

# バージョン情報
__version__ = "2025.09"
__api_version__ = "3.0"

# 互換性情報（破壊的変更）
__breaking_changes__ = [
    "エフェクトは関数ベースに統一 (Geometry -> Geometry)",
    "パイプラインは E.pipeline（単層キャッシュ）",
    "GeometryData を Geometry に統合",
]
