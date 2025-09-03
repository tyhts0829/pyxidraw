#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyxiDraw ベンチマークシステム コマンドライン界面

統一ベンチマークシステムのコマンドライン実行環境。
プラグインシステムと設定管理を統合した使いやすいインターフェース。

使用例:
    python -m benchmarks run                    # 全ベンチマーク実行
    python -m benchmarks run --config custom.yaml
    python -m benchmarks run --parallel        # 並列実行
    python -m benchmarks list                  # 利用可能ターゲット一覧
    python -m benchmarks run --target effects.noise.high_intensity
    python -m benchmarks validate results.json # 結果検証
    python -m benchmarks compare baseline.json current.json
"""

import argparse
import sys
from pathlib import Path

from benchmarks.cli.commands import execute_command
from common.logging import setup_default_logging


def create_parser() -> argparse.ArgumentParser:
    """コマンドラインパーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="PyxiDraw Unified Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python -m benchmarks run                           # 全ベンチマーク実行
  python -m benchmarks run --config custom.yaml     # カスタム設定で実行
  python -m benchmarks run --parallel --workers 4   # 並列実行
  python -m benchmarks run --target effects.noise   # 特定ターゲットのみ
  python -m benchmarks list                         # 利用可能ターゲット一覧
  python -m benchmarks validate results.json        # 結果検証
  python -m benchmarks compare old.json new.json    # 結果比較
  python -m benchmarks config template config.yaml  # 設定テンプレート作成
        """)
    
    # グローバルオプション
    parser.add_argument("--config", "-c", type=Path, 
                       help="設定ファイルパス (YAML/JSON)")
    parser.add_argument("--output-dir", "-o", type=Path,
                       help="出力ディレクトリ")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="詳細出力")
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")
    
    # run サブコマンド
    run_parser = subparsers.add_parser("run", help="ベンチマークを実行")
    run_parser.add_argument("--target", "-t", action="append",
                           help="特定のターゲットを実行 (複数指定可)")
    run_parser.add_argument("--plugin", "-p", action="append",
                           help="特定のプラグインのみ実行")
    run_parser.add_argument("--parallel", action="store_true",
                           help="並列実行を有効化")
    run_parser.add_argument("--workers", type=int,
                           help="並列実行のワーカー数")
    run_parser.add_argument("--warmup", type=int,
                           help="ウォームアップ実行回数")
    run_parser.add_argument("--runs", type=int,
                           help="測定実行回数")
    run_parser.add_argument("--timeout", type=float,
                           help="タイムアウト時間（秒）")
    run_parser.add_argument("--no-save", action="store_true",
                           help="結果を保存しない")
    run_parser.add_argument("--no-charts", action="store_true",
                           help="チャートを生成しない")
    
    # list サブコマンド
    list_parser = subparsers.add_parser("list", help="利用可能なターゲットを表示")
    list_parser.add_argument("--plugin", "-p",
                            help="特定プラグインのターゲットのみ表示")
    list_parser.add_argument("--format", choices=["table", "json", "yaml"],
                            default="table", help="出力フォーマット")
    
    # validate サブコマンド
    validate_parser = subparsers.add_parser("validate", help="ベンチマーク結果を検証")
    validate_parser.add_argument("results_file", type=Path,
                                help="検証する結果ファイル")
    validate_parser.add_argument("--report", type=Path,
                                help="検証レポートの出力先")
    
    # compare サブコマンド
    compare_parser = subparsers.add_parser("compare", help="ベンチマーク結果を比較")
    compare_parser.add_argument("baseline", type=Path,
                               help="ベースライン結果ファイル")
    compare_parser.add_argument("current", type=Path,
                               help="現在の結果ファイル")
    compare_parser.add_argument("--regression-threshold", type=float, default=-0.1,
                               help="回帰検出の閾値 (デフォルト: -10%)")
    
    # config サブコマンド
    config_parser = subparsers.add_parser("config", help="設定管理")
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    template_parser = config_subparsers.add_parser("template", help="設定テンプレートを作成")
    template_parser.add_argument("output_file", type=Path,
                                help="出力ファイルパス")
    
    show_parser = config_subparsers.add_parser("show", help="現在の設定を表示")
    
    return parser


def main() -> int:
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()
    setup_default_logging("DEBUG" if getattr(args, "verbose", False) else "INFO")
    
    if not args.command:
        parser.print_help()
        return 1
    
    # コマンド実行を委譲
    return execute_command(args.command, args)


if __name__ == "__main__":
    sys.exit(main())
