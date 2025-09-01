#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定管理モジュールのテスト
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from benchmarks.core.config import BenchmarkConfigManager
from benchmarks.core.types import BenchmarkConfig, BenchmarkConfigError


class TestBenchmarkConfigManager(unittest.TestCase):
    """BenchmarkConfigManager のテスト"""
    
    def setUp(self):
        """テストの前処理"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = BenchmarkConfigManager()
    
    def tearDown(self):
        """テストの後処理"""
        # 一時ファイルのクリーンアップ
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_default_config(self):
        """デフォルト設定の作成テスト"""
        config = self.config_manager._create_default_config()
        
        self.assertIsInstance(config, BenchmarkConfig)
        self.assertEqual(config.warmup_runs, 5)
        self.assertEqual(config.measurement_runs, 20)
    
    def test_load_nonexistent_config(self):
        """存在しない設定ファイルのテスト"""
        config = self.config_manager.load_config()
        
        # デフォルト設定が返されることを確認
        self.assertIsInstance(config, BenchmarkConfig)
        self.assertEqual(config.warmup_runs, 5)
    
    def test_save_and_load_json_config(self):
        """JSON設定ファイルの保存・読み込みテスト"""
        original_config = BenchmarkConfig(
            warmup_runs=10,
            measurement_runs=30,
            timeout_seconds=45.0,
        )
        
        config_path = Path(self.temp_dir) / "test_config.json"
        self.config_manager.save_config(original_config, config_path, "json")
        
        self.assertTrue(config_path.exists())
        
        loaded_config = self.config_manager.load_config(config_path)
        
        self.assertEqual(loaded_config.warmup_runs, 10)
        self.assertEqual(loaded_config.measurement_runs, 30)
        self.assertEqual(loaded_config.timeout_seconds, 45.0)
    
    @unittest.skipUnless(HAS_YAML, "PyYAML not available")
    def test_save_and_load_yaml_config(self):
        """YAML設定ファイルの保存・読み込みテスト"""
        original_config = BenchmarkConfig(
            warmup_runs=15,
            measurement_runs=40,
            parallel=True,
            max_workers=8,
        )
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        self.config_manager.save_config(original_config, config_path, "yaml")
        
        self.assertTrue(config_path.exists())
        
        loaded_config = self.config_manager.load_config(config_path)
        
        self.assertEqual(loaded_config.warmup_runs, 15)
        self.assertEqual(loaded_config.measurement_runs, 40)
        self.assertTrue(loaded_config.parallel)
        self.assertEqual(loaded_config.max_workers, 8)
    
    def test_validate_config_valid(self):
        """有効な設定の検証テスト"""
        config = BenchmarkConfig(
            warmup_runs=5,
            measurement_runs=20,
            timeout_seconds=30.0,
            output_dir=Path(self.temp_dir),
        )
        
        # 例外が発生しないことを確認
        try:
            self.config_manager.validate_config(config)
        except BenchmarkConfigError:
            self.fail("validate_config raised BenchmarkConfigError unexpectedly")
    
    def test_validate_config_invalid(self):
        """無効な設定の検証テスト"""
        # 負の値を持つ無効な設定
        invalid_config = BenchmarkConfig(
            warmup_runs=-1,
            measurement_runs=0,
            timeout_seconds=-5.0,
        )
        
        with self.assertRaises(BenchmarkConfigError):
            self.config_manager.validate_config(invalid_config)
    
    def test_update_from_env(self):
        """環境変数からの設定更新テスト"""
        # 環境変数を設定
        os.environ["BENCHMARK_WARMUP_RUNS"] = "12"
        os.environ["BENCHMARK_PARALLEL"] = "true"
        os.environ["BENCHMARK_MAX_WORKERS"] = "6"
        
        try:
            original_config = BenchmarkConfig()
            updated_config = self.config_manager.update_from_env(original_config)
            
            self.assertEqual(updated_config.warmup_runs, 12)
            self.assertTrue(updated_config.parallel)
            self.assertEqual(updated_config.max_workers, 6)
        
        finally:
            # 環境変数をクリーンアップ
            for env_var in ["BENCHMARK_WARMUP_RUNS", "BENCHMARK_PARALLEL", "BENCHMARK_MAX_WORKERS"]:
                os.environ.pop(env_var, None)
    
    def test_str_to_bool(self):
        """文字列からbooleanへの変換テスト"""
        self.assertTrue(self.config_manager._str_to_bool("true"))
        self.assertTrue(self.config_manager._str_to_bool("True"))
        self.assertTrue(self.config_manager._str_to_bool("1"))
        self.assertTrue(self.config_manager._str_to_bool("yes"))
        self.assertTrue(self.config_manager._str_to_bool("on"))
        self.assertTrue(self.config_manager._str_to_bool("enabled"))
        
        self.assertFalse(self.config_manager._str_to_bool("false"))
        self.assertFalse(self.config_manager._str_to_bool("False"))
        self.assertFalse(self.config_manager._str_to_bool("0"))
        self.assertFalse(self.config_manager._str_to_bool("no"))
        self.assertFalse(self.config_manager._str_to_bool("off"))
        self.assertFalse(self.config_manager._str_to_bool("disabled"))
    
    def test_create_template_config(self):
        """テンプレート設定の作成テスト"""
        template_path = Path(self.temp_dir) / "template.yaml"
        
        self.config_manager.create_template_config(template_path)
        
        self.assertTrue(template_path.exists())
        
        # テンプレートが正しく読み込めることを確認
        loaded_config = self.config_manager.load_config(template_path)
        self.assertIsInstance(loaded_config, BenchmarkConfig)


if __name__ == "__main__":
    unittest.main()