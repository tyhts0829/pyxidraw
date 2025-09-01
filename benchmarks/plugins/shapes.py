#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形状ベンチマークプラグイン

Geometry対応形状生成のベンチマークを実行するプラグイン。
既存のshapes_benchmark.pyの機能をプラグインアーキテクチャに統合。
"""

from typing import Any, Dict, List, Literal

from benchmarks.core.types import (
    BenchmarkConfig,
    BenchmarkTarget,
    ModuleFeatures,
)
from benchmarks.plugins.base import BenchmarkPlugin, ParametrizedBenchmarkTarget
from benchmarks.plugins.serializable_targets import SerializableShapeTarget


class ShapeBenchmarkPlugin(BenchmarkPlugin):
    """形状生成専用ベンチマークプラグイン"""
    
    @property
    def plugin_type(self) -> str:
        return "shapes"
    
    def discover_targets(self) -> List[BenchmarkTarget]:
        """設定ファイルから形状ターゲットを発見"""
        targets: List[BenchmarkTarget] = []
        variations = self._get_variations()
        for shape_type, shape_variations in variations.items():
            for variation in shape_variations:
                target_name = f"{shape_type}.{variation['name']}"
                params = variation.get('params', {})
                complexity = self._determine_complexity(shape_type, params)
                targets.append(
                    ParametrizedBenchmarkTarget(
                        name=target_name,
                        base_func=SerializableShapeTarget(shape_type, params),
                        parameters=params,
                        shape_type=shape_type,
                        complexity=complexity,
                        metadata={"shape_type": shape_type},
                    )
                )
        return targets
    
    def _determine_complexity(self, shape_type: str, params: dict) -> str:
        """形状タイプとパラメータから複雑さを判定"""
        if shape_type == 'polygon':
            n_sides = params.get('n_sides', 3)
            return "simple" if n_sides <= 6 else "medium" if n_sides <= 20 else "complex"
        elif shape_type == 'sphere':
            subdivisions = params.get('subdivisions', 0)
            return "simple" if subdivisions <= 0.3 else "medium" if subdivisions <= 0.7 else "complex"
        elif shape_type == 'grid':
            rows = params.get('rows', 5)
            cols = params.get('cols', 5)
            total = rows * cols
            return "simple" if total <= 25 else "medium" if total <= 100 else "complex"
        else:
            return "medium"
    
    def create_benchmark_target(self, target_name: str, **kwargs) -> BenchmarkTarget:
        """設定ファイルからカスタムベンチマーク対象を作成"""
        parts = target_name.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid target name format: {target_name}")
        
        shape_type, variation_name = parts
        
        variations = self._get_variations()
        if shape_type in variations:
            for variation in variations[shape_type]:
                if variation['name'] == variation_name:
                    params = variation.get('params', {})
                    complexity = self._determine_complexity(shape_type, params)
                    return ParametrizedBenchmarkTarget(
                        name=target_name,
                        base_func=SerializableShapeTarget(shape_type, params),
                        parameters=params,
                        shape_type=shape_type,
                        complexity=complexity,
                        metadata={"shape_type": shape_type},
                    )
        
        raise ValueError(f"Target not found in config: {target_name}")
    
    def analyze_target_features(self, target: BenchmarkTarget) -> ModuleFeatures:
        """形状対象の特性を分析"""
        features = ModuleFeatures(
            has_njit=False,
            has_cache=True,  # 形状生成にもキャッシュが使用される可能性
            function_count=1,
            source_lines=0,
            import_errors=[]
        )
        
        # 形状タイプに基づく分析
        if hasattr(target, 'metadata'):
            shape_type = target.metadata.get('shape_type', 'unknown')
            
            # 既知の最適化情報
            if shape_type in ['sphere', 'polyhedron']:
                features['has_njit'] = True  # 複雑な形状は数値計算が多い
        
        return features

    # === 内部ユーティリティ ===
    def _get_variations(self) -> Dict[str, List[Dict[str, Any]]]:
        if hasattr(self.config, 'targets') and 'shapes' in getattr(self.config, 'targets', {}):
            shapes_config = self.config.targets['shapes']
            if shapes_config.get('enabled', True):
                return shapes_config.get('variations', {})
        # 組み込みデフォルト（最小限）
        return {
            'polygon': [
                {'name': 'triangle', 'params': {'n_sides': 3}},
            ],
            'sphere': [
                {'name': 'high_res', 'params': {'subdivisions': 1}},
            ],
        }
