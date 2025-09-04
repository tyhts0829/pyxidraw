#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク設定管理モジュール

YAML/JSON設定ファイルの読み込み、環境変数の処理、
設定の検証を行います。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .types import BenchmarkConfig, BenchmarkConfigError, EffectParams


class BenchmarkConfigManager:
    """ベンチマーク設定の管理クラス"""
    
    DEFAULT_CONFIG_PATHS = [
        "benchmarks/config/default.yaml",
        "benchmarks/config/default.json", 
        "benchmark_config.yaml",
        "benchmark_config.json",
    ]
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._config: Optional[BenchmarkConfig] = None
        
    def load_config(self, config_path: Optional[Path] = None) -> BenchmarkConfig:
        """設定ファイルを読み込む"""
        path = config_path or self.config_path
        
        if path is None:
            # デフォルト設定パスを試行
            path = self._find_default_config()
            
        if path is None:
            # デフォルト設定を使用
            return self._create_default_config()
            
        try:
            if not path.exists():
                raise BenchmarkConfigError(f"Config file not found: {path}")
                
            config_data = self._load_file(path)
            return self._parse_config(config_data)
            
        except Exception as e:
            raise BenchmarkConfigError(f"Failed to load config from {path}: {e}")
    
    def save_config(self, config: BenchmarkConfig, path: Path, 
                   format: str = "yaml") -> None:
        """設定をファイルに保存"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self._config_to_dict(config)
            
            if format.lower() == "yaml" and HAS_YAML:
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            elif format.lower() == "json":
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise BenchmarkConfigError(f"Unsupported format: {format}")
                
        except Exception as e:
            raise BenchmarkConfigError(f"Failed to save config to {path}: {e}")
    
    def get_config(self) -> BenchmarkConfig:
        """現在の設定を取得（キャッシュあり）"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> BenchmarkConfig:
        """設定を再読み込み"""
        self._config = None
        return self.get_config()
    
    def update_from_env(self, config: BenchmarkConfig) -> BenchmarkConfig:
        """環境変数から設定を更新"""
        env_mappings = {
            'BENCHMARK_WARMUP_RUNS': ('warmup_runs', int),
            'BENCHMARK_MEASUREMENT_RUNS': ('measurement_runs', int),
            'BENCHMARK_TIMEOUT': ('timeout_seconds', float),
            'BENCHMARK_OUTPUT_DIR': ('output_dir', Path),
            'BENCHMARK_PARALLEL': ('parallel', self._str_to_bool),
            'BENCHMARK_MAX_WORKERS': ('max_workers', int),
            'BENCHMARK_CONTINUE_ON_ERROR': ('continue_on_error', self._str_to_bool),
        }
        
        updated_config = config
        
        for env_var, (attr_name, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    setattr(updated_config, attr_name, converted_value)
                except (ValueError, TypeError) as e:
                    raise BenchmarkConfigError(
                        f"Invalid value for {env_var}: {value} ({e})"
                    )
        
        return updated_config
    
    def validate_config(self, config: BenchmarkConfig) -> None:
        """設定の妥当性をチェック"""
        errors = []
        
        # 基本的な値の検証
        if config.warmup_runs < 0:
            errors.append("warmup_runs must be >= 0")
            
        if config.measurement_runs <= 0:
            errors.append("measurement_runs must be > 0")
            
        if config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be > 0")
            
        if config.max_errors < 0:
            errors.append("max_errors must be >= 0")
            
        if config.parallel and config.max_workers is not None and config.max_workers <= 0:
            errors.append("max_workers must be > 0 when specified")
            
        # ディレクトリの検証
        try:
            config.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {config.output_dir}: {e}")
        
        if errors:
            raise BenchmarkConfigError(f"Invalid configuration: {'; '.join(errors)}")
    
    def create_template_config(self, path: Path) -> None:
        """テンプレート設定ファイルを作成"""
        template = {
            "benchmark": {
                "warmup_runs": 5,
                "measurement_runs": 20,
                "timeout_seconds": 30.0,
                "output_dir": "benchmark_results",
                "continue_on_error": True,
                "max_errors": 10,
                "parallel": False,
                "max_workers": None,
                "generate_charts": True,
                "chart_format": "png",
                "chart_dpi": 150
            },
            "targets": {
                "effects": {
                    "enabled": True,
                    "variations": {
                        "transform": [
                            {"name": "identity", "params": {}},
                            {"name": "scale_2x", "params": {"scale": [2, 2, 2]}},
                            {"name": "rotate_45", "params": {"rotate": [45, 0, 0]}}
                        ],
                        "noise": [
                            {"name": "low", "params": {"intensity": 0.1}},
                            {"name": "medium", "params": {"intensity": 0.5}},
                            {"name": "high", "params": {"intensity": 1.0}}
                        ]
                    }
                },
                "shapes": {
                    "enabled": True,
                    "variations": {
                        "polygon": [
                            {"name": "triangle", "params": {"n_sides": 3}},
                            {"name": "hexagon", "params": {"n_sides": 6}},
                            {"name": "circle", "params": {"n_sides": 32}}
                        ],
                        "sphere": [
                            {"name": "low_res", "params": {"subdivisions": 0}},
                            {"name": "medium_res", "params": {"subdivisions": 0.5}},
                            {"name": "high_res", "params": {"subdivisions": 1}}
                        ]
                    }
                }
            }
        }
        
        self.save_config(self._parse_config(template), path)
    
    def _find_default_config(self) -> Optional[Path]:
        """デフォルト設定ファイルを探す"""
        for config_path in self.DEFAULT_CONFIG_PATHS:
            path = Path(config_path)
            if path.exists():
                return path
        return None
    
    def _load_file(self, path: Path) -> Dict[str, Any]:
        """ファイルを読み込む"""
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml'] and HAS_YAML:
                return yaml.safe_load(f) or {}
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise BenchmarkConfigError(f"Unsupported file format: {path.suffix}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> BenchmarkConfig:
        """設定データをBenchmarkConfigに変換"""
        # 互換性: ルート直下に項目がある場合はそれも受け入れる
        benchmark_section = config_data.get('benchmark', {})
        if not benchmark_section:
            # 旧形式（ルート直下）の場合
            benchmark_section = config_data
        
        # パスの文字列をPathオブジェクトに変換
        output_dir = benchmark_section.get('output_dir', 'benchmark_results')
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        config = BenchmarkConfig(
            warmup_runs=benchmark_section.get('warmup_runs', 5),
            measurement_runs=benchmark_section.get('measurement_runs', 20),
            timeout_seconds=benchmark_section.get('timeout_seconds', 30.0),
            output_dir=output_dir,
            continue_on_error=benchmark_section.get('continue_on_error', True),
            max_errors=benchmark_section.get('max_errors', 10),
            parallel=benchmark_section.get('parallel', False),
            max_workers=benchmark_section.get('max_workers'),
            generate_charts=benchmark_section.get('generate_charts', True),
            chart_format=benchmark_section.get('chart_format', 'png'),
            chart_dpi=benchmark_section.get('chart_dpi', 150),
            abs_threshold=(benchmark_section.get('thresholds', {}) or {}).get('abs_threshold', 0.0),
        )

        # しきい値の詳細（タグ/ターゲット別）
        thresholds = benchmark_section.get('thresholds', {}) or {}
        if isinstance(thresholds, dict):
            by_tag = thresholds.get('by_tag', {}) or {}
            by_target = thresholds.get('by_target', {}) or {}
            if isinstance(by_tag, dict):
                config.regression_threshold_by_tag = {str(k): float(v) for k, v in by_tag.items()}
            if isinstance(by_target, dict):
                config.regression_threshold_by_target = {str(k): float(v) for k, v in by_target.items()}
        
        # targetsセクションを追加
        if 'targets' in config_data:
            config.targets = config_data['targets']
        
        return config
    
    def _config_to_dict(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """BenchmarkConfigを辞書に変換"""
        return {
            "benchmark": {
                "warmup_runs": config.warmup_runs,
                "measurement_runs": config.measurement_runs,
                "timeout_seconds": config.timeout_seconds,
                "output_dir": str(config.output_dir),
                "continue_on_error": config.continue_on_error,
                "max_errors": config.max_errors,
                "parallel": config.parallel,
                "max_workers": config.max_workers,
                "generate_charts": config.generate_charts,
                "chart_format": config.chart_format,
                "chart_dpi": config.chart_dpi,
                "thresholds": {
                    "abs_threshold": getattr(config, 'abs_threshold', 0.0),
                    "by_tag": getattr(config, 'regression_threshold_by_tag', {}),
                    "by_target": getattr(config, 'regression_threshold_by_target', {}),
                },
            }
        }
    
    def _create_default_config(self) -> BenchmarkConfig:
        """デフォルト設定を作成"""
        return BenchmarkConfig()
    
    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """文字列をbooleanに変換"""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')


# シングルトン的な使用のためのグローバル関数
_global_config_manager: Optional[BenchmarkConfigManager] = None


def get_config_manager() -> BenchmarkConfigManager:
    """グローバル設定マネージャーを取得"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = BenchmarkConfigManager()
    return _global_config_manager


def get_config() -> BenchmarkConfig:
    """現在の設定を取得（便利関数）"""
    return get_config_manager().get_config()


def reload_config() -> BenchmarkConfig:
    """設定を再読み込み（便利関数）"""
    return get_config_manager().reload_config()
