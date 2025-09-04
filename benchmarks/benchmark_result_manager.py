#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果管理モジュール

ベンチマーク結果の保存、読み込み、履歴管理を行うクラス。
エラーハンドリング強化版。
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.core.types import BenchmarkResult
from benchmarks.core.exceptions import BenchmarkError

# ロガー設定
logger = logging.getLogger(__name__)


class BenchmarkResultManager:
    """ベンチマーク結果の永続化と管理を行うクラス"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.effects_dir = self.output_dir / "effects"
        
        # ディレクトリ作成（エラーハンドリング付き）
        try:
            self.output_dir.mkdir(exist_ok=True)
            self.effects_dir.mkdir(exist_ok=True)
            logger.info(f"Benchmark result manager initialized: {self.output_dir}")
        except PermissionError as e:
            raise BenchmarkError(f"Permission denied creating output directory {self.output_dir}: {e}")
        except OSError as e:
            raise BenchmarkError(f"Failed to create output directory {self.output_dir}: {e}")
        
        # ディスク容量チェック
        self._check_disk_space()

    def save_results(self, results: Dict[str, BenchmarkResult]) -> str:
        """タイムスタンプ付きでベンチマーク結果を保存"""
        if not results:
            raise ValueError("Cannot save empty results")
        
        try:
            # ディスク容量チェック
            self._check_disk_space()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.effects_dir / f"benchmark_{timestamp}.json"
            
            # 結果をシリアライズ可能な形式に変換
            serializable_results = self._convert_results_for_json(results)
            
            # 一時ファイルに書き込み、その後アトミックに移動
            temp_filename = filename.with_suffix('.tmp')
            
            try:
                with open(temp_filename, "w", encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                
                # アトミックな移動
                temp_filename.replace(filename)
                logger.info(f"Saved benchmark results to {filename}")
                
            except (IOError, OSError) as e:
                # 一時ファイルのクリーンアップ
                if temp_filename.exists():
                    temp_filename.unlink()
                raise BenchmarkError(f"Failed to save results to {filename}: {e}")
            
            # 最新ファイルも安全に更新
            try:
                self._save_latest_file(serializable_results)
            except Exception as e:
                logger.warning(f"Failed to update latest.json: {e}")
                # メインファイルは保存できたので、エラーは警告レベルに留める
            
            return str(filename)
            
        except (PermissionError, OSError) as e:
            raise BenchmarkError(f"Permission denied or I/O error saving results: {e}")
        except json.JSONEncodeError as e:
            raise BenchmarkError(f"Failed to serialize results to JSON: {e}")
        except Exception as e:
            raise BenchmarkError(f"Unexpected error saving results: {e}")

    def load_results(self, filename: str) -> Optional[Dict[str, BenchmarkResult]]:
        """指定されたファイルから結果を読み込む"""
        file_path = Path(filename)
        
        try:
            if not file_path.exists():
                logger.error(f"Benchmark results file not found: {filename}")
                return None
            
            if not file_path.is_file():
                logger.error(f"Path is not a file: {filename}")
                return None
            
            # ファイルサイズチェック（異常に大きなファイルの検出）
            file_size = file_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                logger.warning(f"Large file detected ({file_size / (1024*1024):.1f}MB): {filename}")
            
            with open(file_path, "r", encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    logger.info(f"Loaded benchmark results from {filename}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {filename}: {e}")
                    return None
                    
        except PermissionError as e:
            logger.error(f"Permission denied reading {filename}: {e}")
            return None
        except (IOError, OSError) as e:
            logger.error(f"I/O error reading {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {filename}: {e}")
            return None

    def load_latest_results(self) -> Dict[str, BenchmarkResult]:
        """最新の結果を読み込む"""
        latest_file = self.effects_dir / "latest.json"
        
        if latest_file.exists():
            results = self.load_results(str(latest_file))
            return results if results is not None else {}
        else:
            logger.info("No latest.json file found, returning empty results")
            return {}

    def get_historical_results(self, num_recent: int = 5) -> Dict[str, Dict[str, BenchmarkResult]]:
        """最近のベンチマーク結果を取得"""
        try:
            benchmark_files = sorted(self.effects_dir.glob("benchmark_*.json"))
            
            if not benchmark_files:
                logger.info("No historical benchmark files found")
                return {}
            
            recent_files = benchmark_files[-num_recent:]
            historical_data: Dict[str, Dict[str, BenchmarkResult]] = {}
            
            for file in recent_files:
                try:
                    results = self.load_results(str(file))
                    if results is not None:
                        timestamp = file.stem.replace("benchmark_", "")
                        historical_data[timestamp] = results
                    else:
                        logger.warning(f"Failed to load historical file: {file}")
                except Exception as e:
                    logger.error(f"Error loading historical file {file}: {e}")
                    continue
            
            logger.info(f"Loaded {len(historical_data)} historical benchmark files")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error retrieving historical results: {e}")
            return {}

    def get_all_benchmark_files(self) -> List[Path]:
        """すべてのベンチマークファイルを取得"""
        try:
            files = sorted(self.effects_dir.glob("benchmark_*.json"))
            logger.info(f"Found {len(files)} benchmark files")
            return files
        except Exception as e:
            logger.error(f"Error retrieving benchmark files: {e}")
            return []

    def clean_old_results(self, keep_count: int = 20) -> int:
        """古いベンチマーク結果を削除（最新のkeep_count個を保持）"""
        try:
            benchmark_files = self.get_all_benchmark_files()
            
            if len(benchmark_files) <= keep_count:
                logger.info(f"No cleanup needed, {len(benchmark_files)} files (≤ {keep_count})")
                return 0
            
            files_to_delete = benchmark_files[:-keep_count]
            deleted_count = 0
            
            for file in files_to_delete:
                try:
                    # バックアップディレクトリに移動（完全削除の代わり）
                    backup_dir = self.output_dir / "backup"
                    backup_dir.mkdir(exist_ok=True)
                    
                    backup_path = backup_dir / file.name
                    file.rename(backup_path)
                    deleted_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to delete/backup {file}: {e}")
                    continue
            
            logger.info(f"Cleaned up {deleted_count} old benchmark files")
            logger.info("Moved %d old benchmark files to backup", deleted_count)
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    # === ヘルパーメソッド ===
    
    def _check_disk_space(self, min_free_mb: int = 100) -> None:
        """ディスク容量をチェック"""
        try:
            stat = shutil.disk_usage(self.output_dir)
            free_mb = stat.free / (1024 * 1024)
            
            if free_mb < min_free_mb:
                raise BenchmarkError(f"Insufficient disk space: {free_mb:.1f}MB free (minimum: {min_free_mb}MB)")
            
            logger.debug(f"Disk space check passed: {free_mb:.1f}MB free")
            
        except OSError as e:
            logger.warning(f"Could not check disk space: {e}")
    
    def _convert_results_for_json(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """BenchmarkResultをJSON可能な形式に変換（後方互換なし）"""
        try:
            return {key: self._convert_benchmark_result(result) for key, result in results.items()}
        except Exception as e:
            raise BenchmarkError(f"Failed to convert results for JSON serialization: {e}")
    
    def _convert_benchmark_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """単一のBenchmarkResultを辞書に変換"""
        try:
            return {
                "target_name": result.target_name,
                "plugin_name": result.plugin_name,
                "config": result.config,
                "timestamp": result.timestamp,
                "success": result.success,
                "error_message": result.error_message,
                "schema_version": getattr(result, 'schema_version', '1.0'),
                "timing_data": {
                    "warm_up_times": result.timing_data.warm_up_times,
                    "measurement_times": result.timing_data.measurement_times,
                    "total_time": result.timing_data.total_time,
                    "average_time": result.timing_data.average_time,
                    "std_dev": result.timing_data.std_dev,
                    "min_time": result.timing_data.min_time,
                    "max_time": result.timing_data.max_time
                },
                "metrics": {
                    "vertices_count": result.metrics.vertices_count,
                    "geometry_complexity": result.metrics.geometry_complexity,
                    "memory_usage": result.metrics.memory_usage,
                    "cache_hit_rate": result.metrics.cache_hit_rate
                },
                "output_data": result.output_data,
                "serialization_overhead": result.serialization_overhead,
                "tags": getattr(result, 'tags', []),
                "meta": getattr(result, 'meta', {}),
            }
        except AttributeError as e:
            raise BenchmarkError(f"Invalid BenchmarkResult object structure: {e}")
    
    def _save_latest_file(self, serializable_results: Dict[str, Any]) -> None:
        """最新ファイルを安全に保存"""
        latest_file = self.effects_dir / "latest.json"
        temp_latest = latest_file.with_suffix('.tmp')
        
        try:
            with open(temp_latest, "w", encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            temp_latest.replace(latest_file)
            logger.debug("Updated latest.json successfully")
            
        except Exception:
            if temp_latest.exists():
                temp_latest.unlink()
            raise
    
    def get_storage_info(self) -> Dict[str, Any]:
        """ストレージ情報を取得"""
        try:
            stat = shutil.disk_usage(self.output_dir)
            benchmark_files = self.get_all_benchmark_files()
            
            total_size = sum(f.stat().st_size for f in benchmark_files)
            
            return {
                "output_directory": str(self.output_dir),
                "benchmark_files_count": len(benchmark_files),
                "total_size_mb": total_size / (1024 * 1024),
                "disk_free_mb": stat.free / (1024 * 1024),
                "disk_total_mb": stat.total / (1024 * 1024),
                "disk_used_percent": (stat.used / stat.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {"error": str(e)}
