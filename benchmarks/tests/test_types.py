#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
型定義モジュールのテスト
"""

import unittest
from datetime import datetime
from pathlib import Path

from benchmarks.core.types import (
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkTimeoutError,
    BenchmarkConfigError,
    ModuleDiscoveryError,
    ValidationError,
)


class TestBenchmarkConfig(unittest.TestCase):
    """BenchmarkConfig のテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = BenchmarkConfig()
        
        self.assertEqual(config.warmup_runs, 5)
        self.assertEqual(config.measurement_runs, 20)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.output_dir, Path("benchmark_results"))
        self.assertTrue(config.continue_on_error)
        self.assertEqual(config.max_errors, 10)
        self.assertFalse(config.parallel)
        self.assertIsNone(config.max_workers)
        self.assertTrue(config.generate_charts)
        self.assertEqual(config.chart_format, "png")
        self.assertEqual(config.chart_dpi, 150)
    
    def test_custom_config(self):
        """カスタム設定のテスト"""
        config = BenchmarkConfig(
            warmup_runs=10,
            measurement_runs=50,
            timeout_seconds=60.0,
            output_dir=Path("/tmp/benchmarks"),
            parallel=True,
            max_workers=4,
        )
        
        self.assertEqual(config.warmup_runs, 10)
        self.assertEqual(config.measurement_runs, 50)
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.output_dir, Path("/tmp/benchmarks"))
        self.assertTrue(config.parallel)
        self.assertEqual(config.max_workers, 4)


class TestBenchmarkExceptions(unittest.TestCase):
    """ベンチマーク例外クラスのテスト"""
    
    def test_benchmark_error(self):
        """BenchmarkError のテスト"""
        error = BenchmarkError("Test error", module_name="test_module", error_code="E001")
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.module_name, "test_module")
        self.assertEqual(error.error_code, "E001")
        self.assertIsInstance(error.timestamp, datetime)
    
    def test_benchmark_timeout_error(self):
        """BenchmarkTimeoutError のテスト"""
        error = BenchmarkTimeoutError("Timeout occurred", module_name="slow_module")
        
        self.assertEqual(str(error), "Timeout occurred")
        self.assertEqual(error.module_name, "slow_module")
        self.assertIsInstance(error, BenchmarkError)
    
    def test_benchmark_config_error(self):
        """BenchmarkConfigError のテスト"""
        error = BenchmarkConfigError("Invalid configuration")
        
        self.assertEqual(str(error), "Invalid configuration")
        self.assertIsInstance(error, BenchmarkError)
    
    def test_module_discovery_error(self):
        """ModuleDiscoveryError のテスト"""
        error = ModuleDiscoveryError("Module not found", module_name="missing_module")
        
        self.assertEqual(str(error), "Module not found")
        self.assertEqual(error.module_name, "missing_module")
        self.assertIsInstance(error, BenchmarkError)
    
    def test_validation_error(self):
        """ValidationError のテスト"""
        error = ValidationError("Validation failed")
        
        self.assertEqual(str(error), "Validation failed")
        self.assertIsInstance(error, BenchmarkError)


if __name__ == "__main__":
    unittest.main()