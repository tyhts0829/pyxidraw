#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークシステムテストランナー

すべてのテストを実行し、結果を表示します。
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_tests():
    """すべてのテストを実行"""
    # テストディスカバリーを使用してすべてのテストを検出
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    
    # パターンに一致するテストファイルを検索
    test_suite = loader.discover(
        start_dir=str(test_dir),
        pattern='test_*.py',
        top_level_dir=str(test_dir.parent.parent)
    )
    
    # テスト実行
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 70)
    print("PyxiDraw ベンチマークシステム テスト実行")
    print("=" * 70)
    
    result = runner.run(test_suite)
    
    # 結果の要約を表示
    print("\n" + "=" * 70)
    print("テスト実行結果要約")
    print("=" * 70)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    print(f"スキップ: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    # 失敗やエラーがある場合は詳細を表示
    if result.failures:
        print(f"\n失敗したテスト ({len(result.failures)}件):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nエラーが発生したテスト ({len(result.errors)}件):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # 終了コードを設定
    exit_code = 0 if result.wasSuccessful() else 1
    
    if result.wasSuccessful():
        print("\n✅ すべてのテストが成功しました！")
    else:
        print("\n❌ テストに失敗があります。")
    
    return exit_code


def run_specific_test(test_module: str):
    """特定のテストモジュールを実行"""
    try:
        # テストモジュールを動的にインポート
        module = __import__(f"test_{test_module}", fromlist=[''])
        
        # テストスイートを作成
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # テスト実行
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except ImportError as e:
        print(f"テストモジュール '{test_module}' が見つかりません: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 特定のテストモジュールを実行
        test_module = sys.argv[1]
        exit_code = run_specific_test(test_module)
    else:
        # すべてのテストを実行
        exit_code = run_all_tests()
    
    sys.exit(exit_code)