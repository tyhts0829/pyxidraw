#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
エフェクトベンチマークプラグイン

Geometry対応エフェクトのベンチマークを実行するプラグイン。
既存のeffects_benchmark.pyの機能をプラグインアーキテクチャに統合。
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional

from benchmarks.core.types import (
    BenchmarkConfig,
    BenchmarkTarget,
    ModuleFeatures,
    EffectParams,
)
from benchmarks.core.exceptions import ModuleDiscoveryError, benchmark_operation
from benchmarks.plugins.base import BenchmarkPlugin, ParametrizedBenchmarkTarget
from benchmarks.plugins.serializable_targets import SerializableEffectTarget
from engine.core.geometry import Geometry


class EffectBenchmarkPlugin(BenchmarkPlugin):
    """エフェクト専用ベンチマークプラグイン"""
    
    @property
    def plugin_type(self) -> str:
        return "effects"
    
    def discover_targets(self) -> List[BenchmarkTarget]:
        """設定ファイルからエフェクトターゲットを発見"""
        targets: List[BenchmarkTarget] = []
        # 設定 or 組み込みデフォルトからバリエーションを取得
        variations = self._get_variations()
        for effect_type, effect_variations in variations.items():
            for variation in effect_variations:
                target_name = f"{effect_type}.{variation['name']}"
                params = variation.get('params', {})
                complexity = self._determine_complexity(effect_type, params)
                targets.append(
                    ParametrizedBenchmarkTarget(
                        name=target_name,
                        base_func=SerializableEffectTarget(effect_type, params),
                        parameters=params,
                        effect_type=effect_type,
                        complexity=complexity,
                    )
                )
        return targets
    
    def _determine_complexity(self, effect_type: str, params: dict) -> str:
        """エフェクトタイプとパラメータから複雑さを判定"""
        if effect_type in ['transform', 'scale', 'translate', 'rotate']:
            return "simple"
        elif effect_type in ['noise']:
            frequency = params.get('frequency', 1.0)
            return "complex" if frequency > 2 else "medium"
        elif effect_type in ['subdivision']:
            level = params.get('level', 1)
            return "simple" if level == 1 else "complex" if level >= 3 else "medium"
        elif effect_type in ['array']:
            count_x = params.get('count_x', 1)
            count_y = params.get('count_y', 1)
            return "complex" if count_x * count_y >= 9 else "medium"
        else:
            return "medium"
    
    
    def create_benchmark_target(self, target_name: str, **kwargs) -> BenchmarkTarget:
        """設定ファイルからカスタムベンチマーク対象を作成"""
        parts = target_name.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid target name format: {target_name}")
        
        effect_type, variation_name = parts
        
        variations = self._get_variations()
        if effect_type in variations:
            for variation in variations[effect_type]:
                if variation['name'] == variation_name:
                    params = variation.get('params', {})
                    complexity = self._determine_complexity(effect_type, params)
                    return ParametrizedBenchmarkTarget(
                        name=target_name,
                        base_func=SerializableEffectTarget(effect_type, params),
                        parameters=params,
                        effect_type=effect_type,
                        complexity=complexity,
                    )
        
        raise ValueError(f"Target not found in config: {target_name}")
    
    def analyze_target_features(self, target: BenchmarkTarget) -> ModuleFeatures:
        """エフェクト対象の特性を分析"""
        features = ModuleFeatures(
            has_njit=False,
            has_cache=True,  # Geometryクラスはキャッシュシステムを持つ
            function_count=1,
            source_lines=0,
            import_errors=[]
        )
        
        # エフェクトタイプに基づく分析
        if hasattr(target, 'metadata'):
            effect_type = target.metadata.get('effect_type', 'unknown')
            
            # 既知の最適化情報
            if effect_type in ['transform', 'scale', 'translate', 'rotate']:
                features['has_njit'] = True  # Geometryの変形は高速化されている
            elif effect_type in ['noise', 'subdivision']:
                features['has_njit'] = True  # 数値計算が多い
        
        return features

    # === 内部ユーティリティ ===
    def _get_variations(self) -> Dict[str, List[Dict[str, Any]]]:
        """設定または組み込みデフォルトからエフェクトバリエーションを取得"""
        if hasattr(self.config, 'targets') and 'effects' in getattr(self.config, 'targets', {}):
            effects_config = self.config.targets['effects']
            if effects_config.get('enabled', True):
                return effects_config.get('variations', {})
        # 組み込みデフォルト（最小限、テストに必要なもの）
        return {
            'transform': [
                {'name': 'identity', 'params': {}},
            ],
            'noise': [
                {'name': 'low_intensity', 'params': {'intensity': 0.1, 'frequency': 1.0}},
                {'name': 'high_frequency', 'params': {'intensity': 0.5, 'frequency': 3.0}},
            ],
        }
