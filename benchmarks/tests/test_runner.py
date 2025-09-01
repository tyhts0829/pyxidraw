#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク実行モジュール(runner.py)のテスト
"""

import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.runner import UnifiedBenchmarkRunner
from benchmarks.core.types import BenchmarkResult
from benchmarks.plugins.base import BaseBenchmarkTarget

# --- ヘルパー関数とモッククラス ---

class MockPlugin:
    def __init__(self, name, targets):
        self.name = name
        self._targets = targets

    def get_targets(self, refresh=False):
        return self._targets

    def analyze_target_features(self, target):
        return {"has_njit": False, "has_cache": False, "function_count": 1}

def create_dummy_result(module, success, avg_times):
    return BenchmarkResult(
        module=module, timestamp="", success=success, error=None,
        status="success" if success else "failed",
        timings={k: [v] for k, v in avg_times.items()},
        average_times=avg_times, metrics={}
    )

# --- テストクラス本体 ---

class TestUnifiedBenchmarkRunner(unittest.TestCase):
    """UnifiedBenchmarkRunnerのテスト"""

    def setUp(self):
        self.config = BenchmarkConfig(warmup_runs=1, measurement_runs=2)
        self.runner = UnifiedBenchmarkRunner(config=self.config)

        # シリアライズ可能なGeometryオブジェクトを作成
        from engine.core.geometry import Geometry
        self.test_geometry = Geometry.from_lines([np.zeros((10, 3), dtype=np.float32)])
        
        # シリアライズ可能な関数を使用
        def simple_effect(geom):
            return Geometry.from_lines([geom.coords * 2])
        
        def simple_shape():
            return Geometry.from_lines([np.random.rand(10, 3).astype(np.float32)])
        
        self.effect_target = BaseBenchmarkTarget(name="effects.target1", execute_func=simple_effect)
        self.shape_target = BaseBenchmarkTarget(name="shapes.polygon", execute_func=simple_shape)

        self.runner.plugin_manager = MagicMock()
        self.runner.plugin_manager.get_all_targets.return_value = {
            "effects": [self.effect_target],
            "shapes": [self.shape_target],
        }
        self.runner.plugin_manager.get_all_plugins.return_value = [
            MockPlugin("effects", [self.effect_target]),
            MockPlugin("shapes", [self.shape_target]),
        ]

    @patch("benchmarks.core.runner.UnifiedBenchmarkRunner._benchmark_effect_application", return_value=[0.1, 0.2])
    def test_benchmark_target_isolated_effect(self, mock_benchmark_effect):
        """単一エフェクトターゲットのベンチマークが成功するかのテスト"""
        with patch.object(self.runner, '_get_plugin_for_target', return_value=MockPlugin("effects", [])):
            result = self.runner._benchmark_target_isolated(self.effect_target)
            self.assertTrue(result["success"])
            self.assertAlmostEqual(result["average_times"]["small"], 0.15)

    @patch("benchmarks.core.runner.UnifiedBenchmarkRunner._benchmark_shape_generation", return_value=[0.3, 0.4])
    def test_benchmark_target_isolated_shape(self, mock_benchmark_shape):
        """単一形状ターゲットのベンチマークが成功するかのテスト"""
        with patch.object(self.runner, '_is_shape_target', return_value=True), \
             patch.object(self.runner, '_get_plugin_for_target', return_value=MockPlugin("shapes", [])):
            result = self.runner._benchmark_target_isolated(self.shape_target)
            self.assertTrue(result["success"])
            self.assertAlmostEqual(result["average_times"]["generation"], 0.35)

    def test_is_shape_target(self):
        """ターゲットが形状生成かどうかの判定テスト"""
        # metadataアトリビュートを直接設定
        shape_with_meta = BaseBenchmarkTarget("s", lambda: None)
        shape_with_meta.metadata = {"shape_type": "polygon"}
        self.assertTrue(self.runner._is_shape_target(shape_with_meta))

    @patch("benchmarks.core.runner.UnifiedBenchmarkRunner.benchmark_target")
    def test_run_all_benchmarks_sequential(self, mock_benchmark_target):
        """全ベンチマークの順次実行テスト"""
        self.runner.config.parallel = False
        mock_benchmark_target.return_value = create_dummy_result("mock", True, {"small": 0.1})
        results = self.runner.run_all_benchmarks()
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_benchmark_target.call_count, 2)

    def test_run_all_benchmarks_parallel(self):
        """全ベンチマークの並列実行テスト（実際の並列実行）"""
        self.runner.config.parallel = True
        self.runner.config.max_workers = 1  # テスト環境では1ワーカーに制限
        
        # 実際に並列実行を試みる
        results = self.runner.run_all_benchmarks()
        
        # 結果の確認（実行成功/失敗に関わらず結果が返されることを確認）
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

if __name__ == "__main__":
    unittest.main()