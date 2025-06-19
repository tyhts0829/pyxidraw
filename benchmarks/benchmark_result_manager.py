#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果管理モジュール

ベンチマーク結果の保存、読み込み、履歴管理を行うクラス。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BenchmarkResult = Dict[str, Any]


class BenchmarkResultManager:
    """ベンチマーク結果の永続化と管理を行うクラス"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.effects_dir = self.output_dir / "effects"
        self.effects_dir.mkdir(exist_ok=True)

    def save_results(self, results: Dict[str, BenchmarkResult]) -> str:
        """タイムスタンプ付きでベンチマーク結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.effects_dir / f"benchmark_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        # Also save as latest
        latest_file = self.effects_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2)

        return str(filename)

    def load_results(self, filename: str) -> Dict[str, BenchmarkResult]:
        """指定されたファイルから結果を読み込む"""
        with open(filename, "r") as f:
            return json.load(f)

    def load_latest_results(self) -> Dict[str, BenchmarkResult]:
        """最新の結果を読み込む"""
        latest_file = self.effects_dir / "latest.json"
        if latest_file.exists():
            return self.load_results(str(latest_file))
        return {}

    def get_historical_results(self, num_recent: int = 5) -> Dict[str, Dict[str, BenchmarkResult]]:
        """最近のベンチマーク結果を取得"""
        benchmark_files = sorted(self.effects_dir.glob("benchmark_*.json"))
        
        if not benchmark_files:
            return {}
        
        recent_files = benchmark_files[-num_recent:]
        historical_data: Dict[str, Dict[str, BenchmarkResult]] = {}
        
        for file in recent_files:
            with open(file, "r") as f:
                data = json.load(f)
                timestamp = file.stem.replace("benchmark_", "")
                historical_data[timestamp] = data
        
        return historical_data

    def get_all_benchmark_files(self) -> List[Path]:
        """すべてのベンチマークファイルを取得"""
        return sorted(self.effects_dir.glob("benchmark_*.json"))

    def clean_old_results(self, keep_count: int = 20):
        """古いベンチマーク結果を削除（最新のkeep_count個を保持）"""
        benchmark_files = self.get_all_benchmark_files()
        
        if len(benchmark_files) > keep_count:
            files_to_delete = benchmark_files[:-keep_count]
            for file in files_to_delete:
                file.unlink()
            print(f"Deleted {len(files_to_delete)} old benchmark files")