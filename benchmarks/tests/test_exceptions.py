#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
例外ハンドリングモジュールのテスト
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from benchmarks.core.exceptions import (
    ErrorHandler,
    BenchmarkErrorCollector,
    benchmark_operation,
    get_error_handler,
    get_error_collector,
)
from benchmarks.core.types import (
    BenchmarkError,
    BenchmarkTimeoutError,
    BenchmarkConfigError,
)


class TestErrorHandler(unittest.TestCase):
    """ErrorHandler のテスト"""
    
    def setUp(self):
        """テストの前処理"""
        self.mock_logger = MagicMock(spec=logging.Logger)
        self.error_handler = ErrorHandler(self.mock_logger)
    
    def test_configure(self):
        """設定のテスト"""
        self.error_handler.configure(max_errors=5, continue_on_error=False)
        
        self.assertEqual(self.error_handler.max_errors, 5)
        self.assertFalse(self.error_handler.continue_on_error)
    
    def test_handle_errors_success(self):
        """正常な処理のテスト"""
        with self.error_handler.handle_errors("test_operation"):
            # 何も例外が発生しない場合
            pass
        
        # エラーカウントが増加していないことを確認
        self.assertEqual(self.error_handler.error_count, 0)
    
    def test_handle_errors_with_exception_suppress(self):
        """例外の抑制テスト"""
        self.error_handler.configure(continue_on_error=True)
        
        with self.error_handler.handle_errors("test_operation"):
            raise ValueError("Test error")
        
        # エラーカウントが増加していることを確認
        self.assertEqual(self.error_handler.error_count, 1)
        # ログが呼ばれていることを確認
        self.mock_logger.error.assert_called()
    
    def test_handle_errors_with_exception_no_suppress(self):
        """例外の非抑制テスト"""
        self.error_handler.configure(continue_on_error=False)
        
        with self.assertRaises(BenchmarkError):
            with self.error_handler.handle_errors("test_operation"):
                raise ValueError("Test error")
    
    def test_handle_errors_max_errors_exceeded(self):
        """最大エラー数超過のテスト"""
        self.error_handler.configure(max_errors=2, continue_on_error=True)
        
        # 最大エラー数までエラーを発生させる（エラーカウントが増える）
        for i in range(2):
            try:
                with self.error_handler.handle_errors("test_operation"):
                    raise ValueError(f"Test error {i}")
            except BenchmarkError:
                # エラーカウントが最大値に達した場合の例外は想定内
                pass
        
        # 3回目でBenchmarkErrorが発生することを確認
        with self.assertRaises(BenchmarkError) as cm:
            with self.error_handler.handle_errors("test_operation"):
                raise ValueError("Test error 3")
        
        self.assertIn("Maximum error count", str(cm.exception))
    
    def test_handle_timeout(self):
        """タイムアウトハンドリングのテスト"""
        with self.assertRaises(BenchmarkTimeoutError) as cm:
            self.error_handler.handle_timeout("test_operation", 30.0, "test_module")
        
        error = cm.exception
        self.assertIn("Timeout in test_operation", str(error))
        self.assertEqual(error.module_name, "test_module")
    
    def test_handle_config_error(self):
        """設定エラーハンドリングのテスト"""
        with self.assertRaises(BenchmarkConfigError) as cm:
            self.error_handler.handle_config_error("Invalid config", "/path/to/config")
        
        error = cm.exception
        self.assertIn("Configuration error", str(error))
        self.assertIn("/path/to/config", str(error))
    
    def test_reset_error_count(self):
        """エラーカウントリセットのテスト"""
        self.error_handler.error_count = 5
        self.error_handler.reset_error_count()
        
        self.assertEqual(self.error_handler.error_count, 0)
    
    def test_get_error_summary(self):
        """エラー情報要約のテスト"""
        self.error_handler.configure(max_errors=10, continue_on_error=True)
        self.error_handler.error_count = 3
        
        summary = self.error_handler.get_error_summary()
        
        expected = {
            "error_count": 3,
            "max_errors": 10,
            "continue_on_error": True,
        }
        self.assertEqual(summary, expected)


class TestBenchmarkErrorCollector(unittest.TestCase):
    """BenchmarkErrorCollector のテスト"""
    
    def setUp(self):
        """テストの前処理"""
        self.collector = BenchmarkErrorCollector()
    
    def test_add_error_and_warning(self):
        """エラーと警告の追加テスト"""
        error1 = BenchmarkError("Error 1", module_name="module1")
        error2 = BenchmarkTimeoutError("Timeout", module_name="module2")
        
        self.collector.add_error(error1)
        self.collector.add_error(error2)
        self.collector.add_warning("Warning 1")
        self.collector.add_warning("Warning 2")
        
        self.assertEqual(self.collector.get_error_count(), 2)
        self.assertEqual(self.collector.get_warning_count(), 2)
        self.assertTrue(self.collector.has_errors())
        self.assertTrue(self.collector.has_warnings())
    
    def test_get_errors_by_type(self):
        """エラーの種類別分類テスト"""
        error1 = BenchmarkError("Error 1")
        error2 = BenchmarkTimeoutError("Timeout")
        error3 = BenchmarkError("Error 2")
        
        self.collector.add_error(error1)
        self.collector.add_error(error2)
        self.collector.add_error(error3)
        
        errors_by_type = self.collector.get_errors_by_type()
        
        self.assertEqual(len(errors_by_type["BenchmarkError"]), 2)
        self.assertEqual(len(errors_by_type["BenchmarkTimeoutError"]), 1)
    
    def test_get_errors_by_module(self):
        """エラーのモジュール別分類テスト"""
        error1 = BenchmarkError("Error 1", module_name="module_a")
        error2 = BenchmarkError("Error 2", module_name="module_b")
        error3 = BenchmarkError("Error 3", module_name="module_a")
        
        self.collector.add_error(error1)
        self.collector.add_error(error2)
        self.collector.add_error(error3)
        
        errors_by_module = self.collector.get_errors_by_module()
        
        self.assertEqual(len(errors_by_module["module_a"]), 2)
        self.assertEqual(len(errors_by_module["module_b"]), 1)
    
    def test_generate_report_empty(self):
        """空のレポート生成テスト"""
        report = self.collector.generate_report()
        
        self.assertEqual(report, "No errors or warnings reported.")
    
    def test_generate_report_with_errors_and_warnings(self):
        """エラーと警告を含むレポート生成テスト"""
        error = BenchmarkError("Test error", module_name="test_module")
        self.collector.add_error(error)
        self.collector.add_warning("Test warning")
        
        report = self.collector.generate_report()
        
        self.assertIn("=== ERRORS (1) ===", report)
        self.assertIn("=== WARNINGS (1) ===", report)
        self.assertIn("Test error", report)
        self.assertIn("Test warning", report)
        self.assertIn("[test_module]", report)
    
    def test_clear(self):
        """クリア機能のテスト"""
        self.collector.add_error(BenchmarkError("Error"))
        self.collector.add_warning("Warning")
        
        self.assertTrue(self.collector.has_errors())
        self.assertTrue(self.collector.has_warnings())
        
        self.collector.clear()
        
        self.assertFalse(self.collector.has_errors())
        self.assertFalse(self.collector.has_warnings())
        self.assertEqual(self.collector.get_error_count(), 0)
        self.assertEqual(self.collector.get_warning_count(), 0)


class TestBenchmarkOperation(unittest.TestCase):
    """benchmark_operation コンテキストマネージャーのテスト"""
    
    def test_benchmark_operation_success(self):
        """正常な操作のテスト"""
        with benchmark_operation("test_op", "test_module"):
            # 何も例外が発生しない場合
            pass
        
        # 例外が発生しないことを確認
    
    def test_benchmark_operation_with_error(self):
        """エラーを含む操作のテスト"""
        with patch('benchmarks.core.exceptions.get_error_handler') as mock_handler:
            mock_handler.return_value.handle_errors.return_value.__enter__ = MagicMock()
            mock_handler.return_value.handle_errors.return_value.__exit__ = MagicMock()
            
            with benchmark_operation("test_op", "test_module"):
                pass
            
            # エラーハンドラーが呼ばれることを確認
            mock_handler.assert_called()


class TestGlobalFunctions(unittest.TestCase):
    """グローバル関数のテスト"""
    
    def test_get_error_handler(self):
        """グローバルエラーハンドラーの取得テスト"""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        # 同じインスタンスが返されることを確認（シングルトン）
        self.assertIs(handler1, handler2)
        self.assertIsInstance(handler1, ErrorHandler)
    
    def test_get_error_collector(self):
        """グローバルエラーコレクターの取得テスト"""
        collector1 = get_error_collector()
        collector2 = get_error_collector()
        
        # 同じインスタンスが返されることを確認（シングルトン）
        self.assertIs(collector1, collector2)
        self.assertIsInstance(collector1, BenchmarkErrorCollector)


if __name__ == "__main__":
    unittest.main()