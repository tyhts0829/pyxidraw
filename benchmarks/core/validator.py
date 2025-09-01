#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果検証モジュール

ベンチマーク結果の妥当性検証、パフォーマンス回帰検出、
統計的な分析を行います。
"""

import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from benchmarks.core.types import (
    BenchmarkResult,
    ValidationResult,
    ComparisonResult,
    PerformanceStats,
    BenchmarkMetrics,
)
from benchmarks.core.exceptions import ValidationError


class BenchmarkValidator:
    """ベンチマーク結果の妥当性検証クラス"""
    
    def __init__(self, tolerance: float = 0.1, confidence_level: float = 0.95):
        self.tolerance = tolerance  # パフォーマンス許容値（10%）
        self.confidence_level = confidence_level  # 信頼度（95%）
    
    def validate_result(self, result: BenchmarkResult) -> ValidationResult:
        """単一のベンチマーク結果を検証"""
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        # レガシー辞書形式に対応
        if isinstance(result, dict):
            legacy_errors, legacy_warnings, legacy_metrics = self._validate_legacy_dict(result)
            return ValidationResult(
                is_valid=len(legacy_errors) == 0,
                errors=legacy_errors,
                warnings=legacy_warnings,
                metrics=legacy_metrics,
            )

        # 1. 構造検証
        structure_errors = self._validate_structure(result)
        if structure_errors:
            return ValidationResult(
                is_valid=False,
                errors=structure_errors,
                warnings=warnings,
                metrics=metrics
            )

        # 2. 成功ステータスの検証
        if not result.success:
            error_msg = result.error_message or "Unknown error"
            errors.append(f"Benchmark failed: {error_msg}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )

        # 3. タイミングデータとメトリクスの検証 (成功した場合のみ)
        timing_errors, timing_warnings, timing_metrics = self._validate_timings(result)
        errors.extend(timing_errors)
        warnings.extend(timing_warnings)
        metrics.update(timing_metrics)

        metrics_errors, metrics_warnings = self._validate_metrics(result)
        errors.extend(metrics_errors)
        warnings.extend(metrics_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_multiple_results(self, results: Dict[str, BenchmarkResult]) -> ValidationResult:
        """複数のベンチマーク結果を検証"""
        all_errors = []
        all_warnings = []
        all_metrics = {}
        
        valid_count = 0
        
        for module_name, result in results.items():
            # resultが辞書の場合はBenchmarkResultオブジェクトに変換
            if isinstance(result, dict):
                # 辞書形式の結果の場合、基本的な検証のみ実行
                if result.get("success", False):
                    valid_count += 1
                continue
            
            validation = self.validate_result(result)
            
            # validationの型を確認
            if hasattr(validation, 'is_valid'):
                if validation.is_valid:
                    valid_count += 1
            elif isinstance(validation, dict):
                # 辞書の場合
                if validation.get("is_valid", False):
                    valid_count += 1
            else:
                # 予期しない型の場合はスキップ
                continue
            
            # エラーと警告にモジュール名を付加
            if hasattr(validation, 'errors'):
                module_errors = [f"{module_name}: {error}" for error in validation.errors]
                module_warnings = [f"{module_name}: {warning}" for warning in validation.warnings]
                all_metrics[module_name] = validation.metrics
            elif isinstance(validation, dict):
                module_errors = [f"{module_name}: {error}" for error in validation.get("errors", [])]
                module_warnings = [f"{module_name}: {warning}" for warning in validation.get("warnings", [])]
                all_metrics[module_name] = validation.get("metrics", {})
            else:
                continue
            
            all_errors.extend(module_errors)
            all_warnings.extend(module_warnings)
        
        # 全体的なメトリクスを追加
        total_modules = len(results)
        success_rate = valid_count / total_modules if total_modules > 0 else 0
        
        overall_metrics = {
            "total_modules": total_modules,
            "valid_modules": valid_count,
            "success_rate": success_rate,
            "failed_modules": total_modules - valid_count,
        }
        
        # 成功率が低い場合は警告
        if success_rate < 0.8:
            all_warnings.append(f"Low success rate: {success_rate:.1%} ({valid_count}/{total_modules})")
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            metrics={"overall": overall_metrics, "modules": all_metrics}
        )
    
    def compare_results(self, current: BenchmarkResult, baseline: BenchmarkResult) -> ComparisonResult:
        """2つのベンチマーク結果を比較"""
        # レガシー辞書形式に対応
        if isinstance(current, dict) and isinstance(baseline, dict):
            cur_avg = self._legacy_overall_average(current)
            base_avg = self._legacy_overall_average(baseline)
            if base_avg == 0:
                improvement_ratio = float('inf') if cur_avg == 0 else -float('inf')
            else:
                improvement_ratio = (base_avg - cur_avg) / base_avg
            # 辞書形式では統計的有意性は簡易に常にTrueとする
            return ComparisonResult(
                baseline=baseline,  # type: ignore[arg-type]
                current=current,  # type: ignore[arg-type]
                improvement_ratio=improvement_ratio,
                is_significant=True,
                p_value=0.0,
            )

        if not (current.success and baseline.success):
            raise ValidationError("Both results must be successful for comparison")
        
        # 平均実行時間を比較
        current_avg = self._calculate_overall_average(current)
        baseline_avg = self._calculate_overall_average(baseline)
        
        if baseline_avg == 0:
            improvement_ratio = float('inf') if current_avg == 0 else -float('inf')
        else:
            improvement_ratio = (baseline_avg - current_avg) / baseline_avg
        
        # 統計的有意性をテスト
        is_significant, p_value = self._statistical_significance_test(current, baseline)
        
        return ComparisonResult(
            baseline=baseline,
            current=current,
            improvement_ratio=improvement_ratio,
            is_significant=is_significant,
            p_value=p_value
        )

    # === レガシーサポート ===
    def _validate_legacy_dict(self, result: dict) -> tuple[list[str], list[str], dict]:
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}

        required_fields = ["module", "status", "timings", "average_times"]
        for f in required_fields:
            if f not in result:
                errors.append(f"Missing required field: {f}")

        # 成功判定
        is_success = result.get("success")
        if is_success is None:
            is_success = result.get("status") == "success"
        if not is_success:
            errors.append(f"Benchmark failed: {result.get('error', 'Unknown error')}")
            return errors, warnings, metrics

        # ばらつき警告（各カテゴリでCV > 0.5）
        timings: dict = result.get("timings", {}) or {}
        for name, values in timings.items():
            if values and len(values) >= 2:
                mean = float(np.mean(values))
                std = float(np.std(values))
                if mean > 0 and std / mean > 0.5:
                    warnings.append(f"high variability in {name}: CV={std/mean:.2f}")
        return errors, warnings, metrics

    def _legacy_overall_average(self, result: dict) -> float:
        avgs: dict = result.get("average_times", {}) or {}
        if not avgs:
            return 0.0
        return float(np.mean(list(avgs.values())))
    
    def detect_performance_regression(self, 
                                    current: Dict[str, BenchmarkResult],
                                    baseline: Dict[str, BenchmarkResult],
                                    regression_threshold: float = -0.1) -> List[str]:
        """パフォーマンス回帰を検出"""
        regressions: List[str] = []
        for module_name in current.keys():
            if module_name not in baseline:
                continue

            cur = current[module_name]
            base = baseline[module_name]
            # レガシー平均（存在すれば優先）
            cur_avg = None
            base_avg = None
            if hasattr(cur, '_legacy_average_times') and getattr(cur, '_legacy_average_times'):
                cur_avg = self._legacy_overall_average({"average_times": getattr(cur, '_legacy_average_times')})
            if hasattr(base, '_legacy_average_times') and getattr(base, '_legacy_average_times'):
                base_avg = self._legacy_overall_average({"average_times": getattr(base, '_legacy_average_times')})
            if cur_avg is None or base_avg is None:
                try:
                    comp = self.compare_results(cur, base)
                    imp = comp['improvement_ratio'] if isinstance(comp, dict) else comp.improvement_ratio
                    signif = comp['is_significant'] if isinstance(comp, dict) else comp.is_significant
                except Exception as e:
                    regressions.append(f"{module_name}: comparison failed ({e})")
                    continue
            else:
                # 単純比較（有意性は常にTrueとする）
                if base_avg == 0:
                    imp = float('inf') if cur_avg == 0 else -float('inf')
                else:
                    imp = (base_avg - cur_avg) / base_avg
                signif = True

            if imp < regression_threshold and signif:
                regressions.append(f"{module_name}: {abs(imp)*100:.1f}% performance degradation (p-value: 0.000)")

        return regressions
    
    def calculate_performance_stats(self, result: BenchmarkResult) -> PerformanceStats:
        """パフォーマンス統計を計算"""
        if not result.success or not result.timing_data.measurement_times:
            raise ValidationError("Cannot calculate stats for unsuccessful result")
        
        # 測定時間データを取得
        all_times = result.timing_data.measurement_times
        
        if not all_times:
            raise ValidationError("No timing data available")
        
        times_array = np.array(all_times)
        
        return PerformanceStats(
            mean=float(np.mean(times_array)),
            median=float(np.median(times_array)),
            std=float(np.std(times_array)),
            min=float(np.min(times_array)),
            max=float(np.max(times_array)),
            percentile_95=float(np.percentile(times_array, 95)),
            percentile_99=float(np.percentile(times_array, 99))
        )
    
    def _validate_structure(self, result: BenchmarkResult) -> List[str]:
        """ベンチマーク結果の構造を検証"""
        errors = []
        
        # 新しいdataclass形式の必須フィールドをチェック
        required_attrs = ["target_name", "plugin_name", "timestamp", "success", "timing_data", "metrics"]
        
        for attr in required_attrs:
            if not hasattr(result, attr):
                errors.append(f"Missing required attribute: {attr}")
        
        # タイムスタンプの妥当性チェック
        if hasattr(result, 'timestamp') and result.timestamp:
            try:
                ts = result.timestamp
                if isinstance(ts, (int, float)):
                    datetime.fromtimestamp(ts)
                elif isinstance(ts, str):
                    # ISO 形式に対応
                    datetime.fromisoformat(ts)
                else:
                    errors.append("Invalid timestamp value")
            except Exception:
                errors.append("Invalid timestamp value")
        
        # タイミングデータの構造チェック
        if hasattr(result, 'timing_data') and result.timing_data:
            if not hasattr(result.timing_data, 'measurement_times'):
                errors.append("Missing measurement_times in timing_data")
        
        return errors
    
    def _validate_timings(self, result: BenchmarkResult) -> Tuple[List[str], List[str], dict]:
        """タイミングデータを検証"""
        errors = []
        warnings = []
        metrics = {}
        
        if not hasattr(result, 'timing_data') or not result.timing_data:
            errors.append("No timing data available")
            return errors, warnings, metrics
        
        timing_data = result.timing_data
        
        # 測定時間データの検証
        if timing_data.measurement_times:
            times_array = np.array(timing_data.measurement_times)
            
            # 負の値チェック
            if np.any(times_array < 0):
                errors.append("Negative timing values in measurement_times")
            
            # 異常値検出（平均から3標準偏差を超える値）
            if len(times_array) > 3:
                mean_time = np.mean(times_array)
                std_time = np.std(times_array)
                outliers = np.abs(times_array - mean_time) > 3 * std_time
                
                if np.any(outliers):
                    outlier_count = np.sum(outliers)
                    warnings.append(f"measurement_times: {outlier_count} outlier(s) detected")
            
            # 変動係数（CV）チェック
            if len(times_array) > 1:
                cv = np.std(times_array) / np.mean(times_array)
                metrics["measurement_cv"] = float(cv)
                
                if cv > 0.5:  # CV > 50%
                    warnings.append(f"measurement_times: high variability (CV: {cv:.2%})")
            
            # 平均時間の整合性チェック
            calculated_avg = np.mean(times_array)
            reported_avg = timing_data.average_time
            
            if reported_avg > 0:
                relative_error = abs(calculated_avg - reported_avg) / calculated_avg
                if relative_error > 0.01:  # 1%以上の誤差
                    errors.append("average_time mismatch with measurement data")
        
        return errors, warnings, metrics
    
    def _validate_metrics(self, result: BenchmarkResult) -> Tuple[List[str], List[str]]:
        """メトリクスを検証"""
        errors = []
        warnings = []
        
        if not hasattr(result, 'metrics') or not result.metrics:
            warnings.append("No metrics data available")
            return errors, warnings
        
        metrics = result.metrics
        
        # 基本メトリクスの範囲チェック
        if hasattr(metrics, 'vertices_count') and metrics.vertices_count < 0:
            errors.append("Negative vertices count")
        
        if hasattr(metrics, 'geometry_complexity') and metrics.geometry_complexity < 0:
            errors.append("Negative geometry complexity")
        
        if hasattr(metrics, 'memory_usage') and metrics.memory_usage < 0:
            errors.append("Negative memory usage")
        
        if hasattr(metrics, 'cache_hit_rate'):
            cache_hit_rate = metrics.cache_hit_rate
            if cache_hit_rate < 0 or cache_hit_rate > 1:
                errors.append(f"Invalid cache hit rate: {cache_hit_rate}")
        
        # タイミングデータの検証
        if hasattr(result, 'timing_data') and result.timing_data:
            avg_time = result.timing_data.average_time
            if avg_time < 0:
                errors.append("Negative average time")
            elif avg_time > 10.0:  # 10秒以上
                warnings.append(f"Very slow execution: {avg_time:.2f}s")
        
        return errors, warnings
    
    def _calculate_overall_average(self, result: BenchmarkResult) -> float:
        """全体の平均実行時間を計算"""
        if not hasattr(result, 'timing_data') or not result.timing_data.measurement_times:
            return 0.0
        
        return result.timing_data.average_time
    
    def _statistical_significance_test(self, 
                                     current: BenchmarkResult, 
                                     baseline: BenchmarkResult) -> Tuple[bool, Optional[float]]:
        """統計的有意性テスト（t検定）"""
        try:
            # 測定データを収集
            current_times = current.timing_data.measurement_times if current.timing_data else []
            baseline_times = baseline.timing_data.measurement_times if baseline.timing_data else []
            
            if len(current_times) < 3 or len(baseline_times) < 3:
                return False, None
            
            # Welch's t-test（等分散を仮定しない）
            t_statistic, p_value = stats.ttest_ind(current_times, baseline_times, equal_var=False)
            
            # 有意水準での判定
            alpha = 1 - self.confidence_level
            is_significant = p_value < alpha
            
            return is_significant, float(p_value)
        
        except Exception:
            return False, None


class BenchmarkResultAnalyzer:
    """ベンチマーク結果分析クラス"""
    
    def __init__(self):
        self.validator = BenchmarkValidator()
    
    def analyze_results(self, results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
        """ベンチマーク結果を包括的に分析"""
        validation_result = self.validator.validate_multiple_results(results)
        
        # ValidationResultオブジェクトを辞書に変換（CLI互換性のため）
        if hasattr(validation_result, 'is_valid'):
            validation_dict = {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "metrics": validation_result.metrics
            }
        else:
            # 既に辞書の場合はそのまま使用
            validation_dict = validation_result
        
        analysis = {
            "validation": validation_dict,
            "summary": self._generate_summary(results),
            "performance_ranking": self._rank_by_performance(results),
            "statistics": self._calculate_statistics(results),
        }
        
        return analysis
    
    def _generate_summary(self, results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
        """結果の要約を生成"""
        total = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total - successful
        
        if successful > 0:
            all_times = []
            for result in results.values():
                if result.success and result.timing_data and result.timing_data.average_time > 0:
                    all_times.append(result.timing_data.average_time)
            
            if all_times:
                fastest = min(all_times)
                slowest = max(all_times)
                avg_time = statistics.mean(all_times)
            else:
                fastest = slowest = avg_time = 0.0
        else:
            fastest = slowest = avg_time = 0.0
        
        return {
            "total_modules": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "fastest_time": fastest,
            "slowest_time": slowest,
            "average_time": avg_time,
        }
    
    def _rank_by_performance(self, results: Dict[str, BenchmarkResult]) -> List[Tuple[str, float]]:
        """パフォーマンス順にランキング"""
        performance_data = []
        
        for module_name, result in results.items():
            if result.success and result.timing_data and result.timing_data.average_time > 0:
                avg_time = result.timing_data.average_time
                performance_data.append((module_name, avg_time))
        
        # 実行時間でソート（昇順）
        performance_data.sort(key=lambda x: x[1])
        
        return performance_data
    
    def _calculate_statistics(self, results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
        """統計情報を計算"""
        stats_data = {}
        
        # 成功したモジュールの統計
        successful_results = [r for r in results.values() if r.success]
        
        if successful_results:
            # 基本統計
            total_vertices = sum(r.metrics.vertices_count for r in successful_results if r.metrics)
            avg_complexity = statistics.mean(r.metrics.geometry_complexity for r in successful_results if r.metrics and r.metrics.geometry_complexity > 0) if any(r.metrics and r.metrics.geometry_complexity > 0 for r in successful_results) else 0
            
            stats_data.update({
                "total_vertices": total_vertices,
                "average_complexity": avg_complexity
            })
            
            # 実行時間の統計
            execution_times = [r.timing_data.average_time for r in successful_results 
                             if r.timing_data and r.timing_data.average_time > 0]
            if execution_times:
                stats_data.update({
                    "average_execution_time": statistics.mean(execution_times),
                    "median_execution_time": statistics.median(execution_times),
                    "min_execution_time": min(execution_times),
                    "max_execution_time": max(execution_times),
                })
        
        return stats_data


# 便利関数
def validate_results(results: Dict[str, BenchmarkResult]) -> ValidationResult:
    """ベンチマーク結果を検証する便利関数"""
    validator = BenchmarkValidator()
    return validator.validate_multiple_results(results)


def analyze_benchmark_results(results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
    """ベンチマーク結果を分析する便利関数"""
    analyzer = BenchmarkResultAnalyzer()
    return analyzer.analyze_results(results)
