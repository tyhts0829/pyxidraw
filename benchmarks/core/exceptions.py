#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークシステム例外ハンドリングモジュール

例外の詳細情報取得、ログ記録、エラー回復処理を提供します。
"""

import logging
import traceback
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Type, Union

from .types import (
    BenchmarkError,
    BenchmarkTimeoutError,
    BenchmarkConfigError,
    ModuleDiscoveryError,
    ValidationError,
)


class ErrorHandler:
    """ベンチマークエラーハンドリングクラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_count = 0
        self.max_errors = 10
        self.continue_on_error = True
    
    def configure(self, max_errors: int = 10, continue_on_error: bool = True) -> None:
        """エラーハンドリング設定"""
        self.max_errors = max_errors
        self.continue_on_error = continue_on_error
    
    @contextmanager
    def handle_errors(self, operation: str, module_name: Optional[str] = None,
                     suppress_errors: bool = None) -> Iterator[None]:
        """エラーハンドリングコンテキストマネージャー"""
        if suppress_errors is None:
            suppress_errors = self.continue_on_error
            
        try:
            yield
        except BenchmarkError:
            # 既にベンチマーク例外の場合は再発生
            raise
        except Exception as e:
            # 一般例外をベンチマーク例外に変換
            error_msg = f"Error in {operation}: {str(e)}"
            benchmark_error = BenchmarkError(error_msg, module_name=module_name)
            
            self._log_error(benchmark_error, e)
            self.error_count += 1
            
            if self.error_count >= self.max_errors:
                raise BenchmarkError(
                    f"Maximum error count ({self.max_errors}) exceeded"
                )
            
            if not suppress_errors:
                raise benchmark_error
    
    def handle_timeout(self, operation: str, timeout_seconds: float,
                      module_name: Optional[str] = None) -> None:
        """タイムアウトエラーの処理"""
        error_msg = f"Timeout in {operation} (>{timeout_seconds}s)"
        timeout_error = BenchmarkTimeoutError(error_msg, module_name=module_name)
        self._log_error(timeout_error)
        raise timeout_error
    
    def handle_config_error(self, message: str, config_path: Optional[str] = None) -> None:
        """設定エラーの処理"""
        error_msg = f"Configuration error: {message}"
        if config_path:
            error_msg += f" (config: {config_path})"
        
        config_error = BenchmarkConfigError(error_msg)
        self._log_error(config_error)
        raise config_error
    
    def handle_discovery_error(self, message: str, module_name: str) -> None:
        """モジュール探索エラーの処理"""
        error_msg = f"Module discovery error in {module_name}: {message}"
        discovery_error = ModuleDiscoveryError(error_msg, module_name=module_name)
        self._log_error(discovery_error)
        raise discovery_error
    
    def handle_validation_error(self, message: str, data: Optional[Any] = None) -> None:
        """検証エラーの処理"""
        error_msg = f"Validation error: {message}"
        if data is not None:
            error_msg += f" (data: {data})"
        
        validation_error = ValidationError(error_msg)
        self._log_error(validation_error)
        raise validation_error
    
    def reset_error_count(self) -> None:
        """エラーカウントをリセット"""
        self.error_count = 0
    
    def get_error_summary(self) -> dict:
        """エラー情報の要約を取得"""
        return {
            "error_count": self.error_count,
            "max_errors": self.max_errors,
            "continue_on_error": self.continue_on_error,
        }
    
    def _log_error(self, error: BenchmarkError, 
                   original_exception: Optional[Exception] = None) -> None:
        """エラーをログに記録"""
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "module_name": getattr(error, 'module_name', None),
            "error_code": getattr(error, 'error_code', None),
            "timestamp": getattr(error, 'timestamp', None),
        }
        
        if original_exception:
            error_info["original_exception"] = type(original_exception).__name__
            error_info["traceback"] = traceback.format_exc()
        
        self.logger.error(f"Benchmark error: {error_info}")


class BenchmarkErrorCollector:
    """ベンチマークエラーの収集と分析クラス"""
    
    def __init__(self):
        self.errors: list[BenchmarkError] = []
        self.warnings: list[str] = []
    
    def add_error(self, error: BenchmarkError) -> None:
        """エラーを追加"""
        self.errors.append(error)
    
    def add_warning(self, message: str) -> None:
        """警告を追加"""
        self.warnings.append(message)
    
    def has_errors(self) -> bool:
        """エラーがあるかチェック"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """警告があるかチェック"""
        return len(self.warnings) > 0
    
    def get_error_count(self) -> int:
        """エラー数を取得"""
        return len(self.errors)
    
    def get_warning_count(self) -> int:
        """警告数を取得"""
        return len(self.warnings)
    
    def get_errors_by_type(self) -> dict[str, list[BenchmarkError]]:
        """エラーを種類別に分類"""
        errors_by_type: dict[str, list[BenchmarkError]] = {}
        
        for error in self.errors:
            error_type = type(error).__name__
            if error_type not in errors_by_type:
                errors_by_type[error_type] = []
            errors_by_type[error_type].append(error)
        
        return errors_by_type
    
    def get_errors_by_module(self) -> dict[str, list[BenchmarkError]]:
        """エラーをモジュール別に分類"""
        errors_by_module: dict[str, list[BenchmarkError]] = {}
        
        for error in self.errors:
            module_name = getattr(error, 'module_name', 'unknown')
            if module_name not in errors_by_module:
                errors_by_module[module_name] = []
            errors_by_module[module_name].append(error)
        
        return errors_by_module
    
    def generate_report(self) -> str:
        """エラーレポートを生成"""
        if not self.has_errors() and not self.has_warnings():
            return "No errors or warnings reported."
        
        report_lines = []
        
        if self.has_errors():
            report_lines.append(f"=== ERRORS ({self.get_error_count()}) ===")
            
            errors_by_type = self.get_errors_by_type()
            for error_type, error_list in errors_by_type.items():
                report_lines.append(f"\n{error_type} ({len(error_list)}):")
                for error in error_list:
                    module_info = f" [{error.module_name}]" if error.module_name else ""
                    report_lines.append(f"  - {error}{module_info}")
        
        if self.has_warnings():
            report_lines.append(f"\n=== WARNINGS ({self.get_warning_count()}) ===")
            for warning in self.warnings:
                report_lines.append(f"  - {warning}")
        
        return "\n".join(report_lines)
    
    def clear(self) -> None:
        """すべてのエラーと警告をクリア"""
        self.errors.clear()
        self.warnings.clear()


# グローバルエラーハンドラー
_global_error_handler: Optional[ErrorHandler] = None
_global_error_collector: Optional[BenchmarkErrorCollector] = None


def get_error_handler() -> ErrorHandler:
    """グローバルエラーハンドラーを取得"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def get_error_collector() -> BenchmarkErrorCollector:
    """グローバルエラーコレクターを取得"""
    global _global_error_collector
    if _global_error_collector is None:
        _global_error_collector = BenchmarkErrorCollector()
    return _global_error_collector


def configure_error_handling(max_errors: int = 10, 
                           continue_on_error: bool = True) -> None:
    """グローバルエラーハンドリング設定"""
    get_error_handler().configure(max_errors, continue_on_error)


@contextmanager
def benchmark_operation(operation: str, module_name: Optional[str] = None,
                       suppress_errors: bool = None) -> Iterator[None]:
    """ベンチマーク操作のエラーハンドリング（便利関数）"""
    with get_error_handler().handle_errors(operation, module_name, suppress_errors):
        yield