"""
CLI コマンド実装

各CLIコマンドの実装を責務別に分離したモジュール
"""
import json
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from benchmarks.core.config import BenchmarkConfigManager
from benchmarks.core.runner import UnifiedBenchmarkRunner
from benchmarks.core.validator import BenchmarkValidator, BenchmarkResultAnalyzer
from benchmarks.benchmark_result_manager import BenchmarkResultManager
from benchmarks.core.types import BenchmarkResult


class CommandExecutor:
    """コマンド実行の基底クラス"""
    
    def __init__(self, args):
        self.args = args
        self.verbose = getattr(args, 'verbose', False)
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """コマンドライン引数から設定を読み込み"""
        config_manager = BenchmarkConfigManager(self.args.config)
        config = config_manager.load_config()
        
        # コマンドライン引数で設定を上書き
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            config.output_dir = self.args.output_dir
        
        if hasattr(self.args, 'parallel') and self.args.parallel:
            config.parallel = True
        
        if hasattr(self.args, 'workers') and self.args.workers:
            config.max_workers = self.args.workers
        
        if hasattr(self.args, 'warmup') and self.args.warmup:
            config.warmup_runs = self.args.warmup
        
        if hasattr(self.args, 'runs') and self.args.runs:
            config.measurement_runs = self.args.runs
        
        if hasattr(self.args, 'timeout') and self.args.timeout:
            config.timeout_seconds = self.args.timeout
        
        if hasattr(self.args, 'no_charts') and self.args.no_charts:
            config.generate_charts = False
        
        # 設定の妥当性をチェック
        config_manager.validate_config(config)
        
        return config
    
    def print_verbose(self, message: str):
        """詳細出力"""
        if self.verbose:
            self.logger.debug(message)


class RunCommand(CommandExecutor):
    """runコマンドの実装"""
    
    def execute(self) -> int:
        """runコマンドを実行"""
        config = self.load_config()
        self.print_verbose(f"設定: {config}")
        
        # ランナーを作成
        runner = UnifiedBenchmarkRunner(config)
        
        try:
            # ベンチマーク実行
            results = self._run_benchmarks(runner)
            
            if not results:
                self.logger.warning("No benchmarks were executed")
                return 1
            
            # 結果の保存
            self._save_results(results, config)
            
            # 結果の分析と表示
            self._analyze_and_display_results(results)
            
            return 0
            
        except Exception as e:
            self.logger.exception("Error running benchmarks: %s", e)
            return 1
    
    def _run_benchmarks(self, runner: UnifiedBenchmarkRunner) -> List[BenchmarkResult]:
        """ベンチマークを実行"""
        # --from-file の読み込み
        targets_from_file: List[str] = []
        if getattr(self.args, 'from_file', None):
            try:
                with open(self.args.from_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        targets_from_file = [str(x) for x in data]
                    elif isinstance(data, dict) and 'targets' in data:
                        targets_from_file = [str(x) for x in data['targets']]
            except Exception as e:
                self.logger.warning("Failed to read --from-file: %s", e)

        # 優先順: --target / --from-file / all
        if getattr(self.args, 'target', None):
            names: List[str] = self.args.target
        elif targets_from_file:
            names = targets_from_file
        else:
            names = []

        # スキップ指定
        skip: set[str] = set(getattr(self.args, 'skip', []) or [])

        if names:
            # 指定ターゲットからスキップを除外
            names = [n for n in names if n not in skip]
            results_dict = runner.run_specific_targets(names)
            return list(results_dict.values())
        else:
            # 全実行 → スキップは後段でフィルタ
            all_targets = runner.plugin_manager.get_all_targets()
            # スキップ対象を除外
            for plugin, targets in all_targets.items():
                all_targets[plugin] = [t for t in targets if t.name not in skip and f"{plugin}.{t.name}" not in skip]
            # 実行
            results: List[BenchmarkResult] = []
            for plugin, targets in all_targets.items():
                if not targets:
                    continue
                # 一時ランナーで逐次実行（既存run_all_benchmarksは再利用しない）
                # ただし既存のrun_all_benchmarksでもスキップ影響を出すには内部変更が必要なため簡易実装
                for t in targets:
                    res = runner.benchmark_target(t)
                    results.append(res)
            return results
    
    def _save_results(self, results: List[BenchmarkResult], config):
        """結果を保存"""
        if not getattr(self.args, 'no_save', False):
            # リストを辞書形式に変換
            results_dict = {result.target_name: result for result in results}
            result_manager = BenchmarkResultManager(str(config.output_dir))
            saved_file = result_manager.save_results(results_dict)
            self.logger.info("Results saved to: %s", saved_file)
            # 失敗ターゲットを書き出し
            failed = [r.target_name for r in results if not r.success]
            try:
                fail_path = Path(config.output_dir) / "failed_targets.json"
                with open(fail_path, 'w', encoding='utf-8') as f:
                    json.dump(failed, f, indent=2, ensure_ascii=False)
                self.logger.info("Failed targets saved to: %s", fail_path)
            except Exception as e:
                self.logger.warning("Failed to save failed_targets.json: %s", e)
    
    def _analyze_and_display_results(self, results: List[BenchmarkResult]):
        """結果を分析して表示"""
        try:
            analyzer = BenchmarkResultAnalyzer()
            # リストを辞書形式に変換
            results_dict = {result.target_name: result for result in results}
            analysis = analyzer.analyze_results(results_dict)
            
            # 要約を表示
            self._display_summary(analysis["summary"])
            
            # 検証結果を表示
            validation = analysis["validation"]
            self._display_validation(validation)
        except Exception as e:
            self.logger.exception("Error in result analysis: %s", e)
            import traceback
            traceback.print_exc()
            raise
    
    def _display_summary(self, summary: Dict[str, Any]):
        """要約を表示"""
        print(f"\n=== BENCHMARK SUMMARY ===")
        print(f"Total modules: {summary['total_modules']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        if summary['successful'] > 0:
            print(f"Fastest time: {summary['fastest_time']*1000:.3f}ms")
            print(f"Slowest time: {summary['slowest_time']*1000:.3f}ms")
            print(f"Average time: {summary['average_time']*1000:.3f}ms")
    
    def _display_validation(self, validation: Dict[str, Any]):
        """検証結果を表示"""
        if validation["errors"]:
            print(f"\n⚠️  Validation errors: {len(validation['errors'])}")
            for error in validation["errors"][:5]:  # 最初の5個のみ表示
                print(f"  - {error}")
            if len(validation["errors"]) > 5:
                print(f"  ... and {len(validation['errors']) - 5} more")


class ListCommand(CommandExecutor):
    """listコマンドの実装"""
    
    def execute(self) -> int:
        """listコマンドを実行"""
        try:
            config = self.load_config()
            runner = UnifiedBenchmarkRunner(config)
            # ターゲットを取得（タグフィルタのため詳細取得）
            all_targets = runner.plugin_manager.get_all_targets()
            filtered: List[str] = []
            for plugin_name, tlist in all_targets.items():
                for t in tlist:
                    fq = f"{plugin_name}.{t.name}"
                    filtered.append(fq)
            targets = filtered
            # プラグインフィルタ
            if getattr(self.args, 'plugin', None):
                targets = [t for t in targets if t.startswith(self.args.plugin)]
            # タグフィルタ
            if getattr(self.args, 'tag', None):
                want_tags = set(self.args.tag)
                # 再取得してタグで落とす
                tagged: List[str] = []
                for plugin_name, tlist in all_targets.items():
                    for t in tlist:
                        t_tags = set(getattr(t, 'tags', []) or [])
                        if want_tags.issubset(t_tags):
                            tagged.append(f"{plugin_name}.{t.name}")
                targets = [t for t in targets if t in tagged]
            
            # フォーマット別出力
            format_type = getattr(self.args, 'format', 'table')
            self._display_targets(targets, format_type)
            
            return 0
            
        except Exception as e:
            self.logger.exception("Error listing targets: %s", e)
            return 1
    
    def _display_targets(self, targets: List[str], format_type: str):
        """ターゲットを指定フォーマットで表示"""
        if format_type == "json":
            print(json.dumps(targets, indent=2))
        elif format_type == "yaml":
            print("targets:")
            for target in targets:
                print(f"  - {target}")
        else:  # table
            print("Available benchmark targets:")
            for target in targets:
                print(f"  {target}")


class ValidateCommand(CommandExecutor):
    """validateコマンドの実装"""
    
    def execute(self) -> int:
        """validateコマンドを実行"""
        try:
            results_file = self.args.results_file
            
            if not results_file.exists():
                self.logger.error("Results file not found: %s", results_file)
                return 1
            
            # 結果を読み込み
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            # 検証実行
            validator = BenchmarkValidator()
            validation_result = validator.validate_multiple_results(results_data)
            
            # ValidationResultオブジェクトを辞書に変換
            validation_dict = {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "metrics": validation_result.metrics
            }
            
            # レポート生成
            if hasattr(self.args, 'report') and self.args.report:
                self._generate_validation_report(validation_dict, self.args.report)
            else:
                self._display_validation_result(validation_dict)
            
            return 0 if validation_result.is_valid else 1
            
        except Exception as e:
            self.logger.exception("Error validating results: %s", e)
            return 1
    
    def _display_validation_result(self, result: Dict[str, Any]):
        """検証結果を表示"""
        print(f"Validation result: {'PASS' if result['is_valid'] else 'FAIL'}")
        
        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result["errors"]:
                print(f"  - {error}")
        
        if result["warnings"]:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result["warnings"]:
                print(f"  - {warning}")
    
    def _generate_validation_report(self, result: Dict[str, Any], report_path: Path):
        """検証レポートを生成"""
        report_content = {
            "validation_status": "PASS" if result["is_valid"] else "FAIL",
            "timestamp": result.get("timestamp"),
            "errors": result["errors"],
            "warnings": result["warnings"],
            "statistics": result.get("statistics", {})
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_content, f, indent=2)
        
        self.logger.info("Validation report saved to: %s", report_path)


class CompareCommand(CommandExecutor):
    """compareコマンドの実装"""
    
    def execute(self) -> int:
        """compareコマンドを実行"""
        try:
            baseline_file = self.args.baseline
            current_file = self.args.current
            
            if not baseline_file.exists():
                self.logger.error("Baseline file not found: %s", baseline_file)
                return 1
            
            if not current_file.exists():
                self.logger.error("Current file not found: %s", current_file)
                return 1
            
            # 結果を読み込み
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            with open(current_file, 'r') as f:
                current_data = json.load(f)
            
            # タグフィルタ（保存結果に 'tags' がある前提。なければ通さない）
            if getattr(self.args, 'tag', None):
                need = set(self.args.tag)
                baseline_data = {k: v for k, v in baseline_data.items() if isinstance(v, dict) and need.issubset(set(v.get('tags', []) or []))}
                current_data = {k: v for k, v in current_data.items() if isinstance(v, dict) and need.issubset(set(v.get('tags', []) or []))}

            # 比較実行（辞書同士の軽量比較）
            comparison = self._compare_dicts(baseline_data, current_data)

            # 結果表示
            self._display_comparison_result(comparison)

            # 回帰検出（タグ/ターゲット別閾値と絶対値閾値を考慮）
            cfg = self.load_config()
            threshold_default = getattr(self.args, 'regression_threshold', -0.1)
            abs_thr = getattr(self.args, 'abs_threshold', 0.0) or 0.0
            regressions = []
            for c in comparison:
                tgt = c["target"]
                change = c.get("performance_change", 0.0)
                abs_diff = c.get("abs_diff", 0.0)
                if abs_diff < abs_thr:
                    continue  # 無視（ノイズ）
                # ターゲット/タグ別のしきい値があれば上書き
                thr = cfg.regression_threshold_by_target.get(tgt, threshold_default)
                # タグはファイルのcurrent側を参照（なければbaseline）
                tags = []
                if tgt in current_data and isinstance(current_data[tgt], dict):
                    tags = current_data[tgt].get('tags', []) or []
                elif tgt in baseline_data and isinstance(baseline_data[tgt], dict):
                    tags = baseline_data[tgt].get('tags', []) or []
                for t in tags:
                    if t in cfg.regression_threshold_by_tag:
                        thr = cfg.regression_threshold_by_tag[t]
                        break
                if change < thr:
                    regressions.append(c)
            
            if regressions:
                self.logger.warning("⚠️  Performance regressions detected: %d", len(regressions))
                return 1
            
            return 0
            
        except Exception as e:
            self.logger.exception("Error comparing results: %s", e)
            return 1
    
    def _display_comparison_result(self, comparison: List[Dict[str, Any]]):
        """比較結果を表示"""
        print("=== BENCHMARK COMPARISON ===")
        
        for result in comparison:
            target = result["target"]
            change = result.get("performance_change", 0)
            abs_diff = result.get("abs_diff", 0.0)
            
            if change > 0.05:  # 5%以上の改善
                status = "🚀 IMPROVED"
            elif change < -0.05:  # 5%以上の劣化
                status = "🐌 REGRESSED"
            else:
                status = "➡️  UNCHANGED"
            
            print(f"{target}: {status} ({change:+.1%}, Δ={abs_diff*1000:.3f} ms)")

    def _compare_dicts(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> List[Dict[str, Any]]:
        """JSON辞書（ファイル読み込み結果）同士の比較を行い、ターゲットごとの差分を返す。"""
        results: List[Dict[str, Any]] = []
        keys = sorted(set(baseline.keys()) & set(current.keys()))
        for k in keys:
            b = baseline[k]
            c = current[k]
            # 平均時間の抽出（新/旧の両形式に対応）
            def _avg(x: Any) -> float:
                # 新フォーマットのみ対応
                if isinstance(x, dict) and isinstance(x.get('timing_data'), dict):
                    return float(x['timing_data'].get('average_time', 0.0) or 0.0)
                return 0.0
            b_avg = _avg(b)
            c_avg = _avg(c)
            if b_avg == 0:
                change = float('inf') if c_avg == 0 else -float('inf')
            else:
                change = (b_avg - c_avg) / b_avg
            results.append({
                "target": k,
                "baseline_avg": b_avg,
                "current_avg": c_avg,
                "performance_change": change,
                "abs_diff": abs(c_avg - b_avg),
            })
        return results


class ConfigCommand(CommandExecutor):
    """configコマンドの実装"""
    
    def execute(self) -> int:
        """configコマンドを実行"""
        try:
            action = getattr(self.args, 'config_action', None)
            
            if action == 'template':
                return self._create_template()
            elif action == 'show':
                return self._show_config()
            else:
                self.logger.error("No config action specified")
                return 1
                
        except Exception as e:
            self.logger.exception("Error in config command: %s", e)
            return 1
    
    def _create_template(self) -> int:
        """設定テンプレートを作成"""
        output_file = self.args.output_file
        config_manager = BenchmarkConfigManager()
        
        template = config_manager.generate_template()
        
        with open(output_file, 'w') as f:
            f.write(template)
        
        self.logger.info("Configuration template created: %s", output_file)
        return 0
    
    def _show_config(self) -> int:
        """現在の設定を表示"""
        config = self.load_config()
        self.logger.info("Current configuration:")
        print(json.dumps(config.__dict__, indent=2, default=str))
        return 0


# コマンド登録マップ
COMMAND_MAP = {
    'run': RunCommand,
    'list': ListCommand,
    'validate': ValidateCommand,
    'compare': CompareCommand,
    'config': ConfigCommand,
}


def execute_command(command: str, args) -> int:
    """指定されたコマンドを実行"""
    if command not in COMMAND_MAP:
        logging.getLogger(__name__).error("Unknown command: %s", command)
        return 1
    
    command_class = COMMAND_MAP[command]
    executor = command_class(args)
    return executor.execute()
