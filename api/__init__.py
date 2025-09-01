"""
PyxiDraw 次期API（2025-06版）

形状チェーンとエフェクトチェーンを分離し、キャッシュと拡張性を両立。

Usage:
    from api import G, E
    
    # 形状チェーン
    sphere = (G.sphere(subdivisions=0.5)
                .size(100, 100, 100)
                .at(100, 100, 0))
    
    # エフェクトチェーン
    result = (E.add(sphere)
                .noise(intensity=0.3)
                .filling(density=0.5)
                .result())
"""

# 主要なAPIクラスをエクスポート
from .geometry_api import GeometryAPI
from .shape_factory import G, ShapeFactory
from .effect_chain import E, EffectFactory, EffectChain

# コアクラスもエクスポート（高度な使用のため）
from engine.core.geometry_data import GeometryData

__all__ = [
    # メインAPI
    "G",           # 形状ファクトリ
    "E",           # エフェクトファクトリ
    
    # クラス（高度な使用）
    "GeometryAPI", 
    "ShapeFactory",
    "EffectFactory",
    "EffectChain",
    "GeometryData",
]

# バージョン情報
__version__ = "2025.06"
__api_version__ = "2.0"

# 互換性情報
__breaking_changes__ = [
    "形状生成APIがG.*に変更",
    "エフェクトチェーンがE.add().*に変更", 
    "GeometryAPIによるメソッドチェーン形式",
    "後方互換性なし（完全リファクタリング）"
]