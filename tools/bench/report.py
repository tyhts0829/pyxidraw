"""
ベンチマーク一発実行 CLI。

以下の 3 ステップを順に実行して `benchmark_results/index.html` を更新する。

1) pytest（perf マーカー）実行 + JSON 出力（perf.json）
2) JSON 要約（summary.json）
3) HTML レポート生成（index.html）

使い方:
    python -m tools.bench.report

備考:
- 追加引数は受け付けない（シンプルさ優先）。必要になれば拡張する。
- 依存: pytest, pytest-benchmark。未導入の場合はエラー表示して終了。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str]) -> int:
    """サブプロセスを実行して終了コードを返す。"""
    try:
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"コマンド失敗: {' '.join(map(str, e.cmd))} (exit {e.returncode})")
        return int(e.returncode or 1)


def main(argv: Optional[List[str]] = None) -> int:
    cwd = Path.cwd()
    json_path = cwd / "perf.json"
    out_dir = cwd / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) pytest（perf マーカーのみ）+ JSON 出力
    step1 = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/perf",
        "-m",
        "perf",
        "--benchmark-only",
        f"--benchmark-json={json_path}",
    ]
    print("[1/3] pytest (perf) 実行中…")
    rc = _run(step1)
    if rc != 0:
        print("ヒント: dev 依存を導入してください（pip install -e .[dev]）。")
        return rc

    # 2) JSON 要約（summary.json）
    summary_json = out_dir / "summary.json"
    step2 = [
        sys.executable,
        "-m",
        "tools.bench.summarize",
        str(json_path),
        "--out-json",
        str(summary_json),
    ]
    print("[2/3] 要約 (summary.json) 生成中…")
    rc = _run(step2)
    if rc != 0:
        return rc

    # 3) HTML レポート（index.html）
    index_html = out_dir / "index.html"
    step3 = [
        sys.executable,
        "-m",
        "tools.bench.html_report",
        str(summary_json),
        "--out",
        str(index_html),
    ]
    print("[3/3] HTML レポート生成中…")
    rc = _run(step3)
    if rc != 0:
        return rc

    print(f"完了: {index_html}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI エントリ
    raise SystemExit(main())
