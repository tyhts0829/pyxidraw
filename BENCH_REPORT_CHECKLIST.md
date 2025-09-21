# ベンチレポート一発生成: 提案とチェックリスト

目的: `pytest` 実行から `benchmark_results/index.html` 更新までを「一発」で実行できる手段を用意する。

即時ワンライナー（現状でも可）:

```sh
pytest -q tests/perf -m perf --benchmark-only --benchmark-json=perf.json \
  && python -m tools.bench.summarize perf.json --out-json benchmark_results/summary.json \
  && python -m tools.bench.html_report benchmark_results/summary.json --out benchmark_results/index.html
```

## 選択肢（提案）

- 1) Makefile のターゲット追加（例: `make perf-report`）
  - 長所: 最小・依存ゼロ。コマンドが短い。
  - 短所: Windows（Make 未導入環境）でそのまま使えない。

- 2) シェルスクリプト追加（`tools/bench/report.sh`）
  - 長所: 依存ゼロ、わかりやすい。`./tools/bench/report.sh` で実行。
  - 短所: Windows 互換性が弱い（WSL/Git Bash 前提）。

- 3) Python CLI 追加（`tools/bench/report.py`）
  - 長所: クロスプラットフォーム。引数/拡張が容易（`--out-dir` など）。
  - 短所: 小さなコード追加が必要。

- 4) コンソールスクリプト公開（`pyproject.toml [project.scripts]`）
  - 長所: `pxd-bench-report` のように 1 コマンド化できる。
  - 短所: `pip install -e .` 済みが前提。スクリプト 3) とセット運用推奨。

推奨: まず 3) Python CLI を追加し、必要なら 4) で 1 コマンド化。開発環境では 1) も併設すると便利。

## 実装チェックリスト

- [x] 方式の確定（3: Python CLI）
- [x] 実装（選択肢に応じて以下）
  - [ ] 1) Makefile: `perf-report` ターゲット追加（上記ワンライナー）
  - [ ] 2) `tools/bench/report.sh`: `set -euo pipefail`、出力ディレクトリ作成、3 コマンド連続実行
  - [x] 3) `tools/bench/report.py`: `subprocess.run` で 3 段を実行（引数は当面なし）
  - [ ] 4) `pyproject.toml` に `[project.scripts]` で `pxd-bench-report = "tools.bench.report:main"`
- [ ] README（または AGENTS.md）に使い方を 1〜2 行追記（未）
- [ ] 変更ファイルの ruff/black/isort/mypy を通す（対象限定）

## 仕様（3) Python CLI 案の叩き台）

- コマンド: `python -m tools.bench.report`
- 動作: 
  1. pytest を実行して `--benchmark-json` で JSON 生成
  2. `tools.bench.summarize` で `summary.json` 作成
  3. `tools.bench.html_report` で `index.html` 作成
- 失敗時: その段で終了コード非 0 を返す。ネットワーク動作や依存の自動導入は行わない。

## 確認したいこと（ご回答ください）

1. 採用する方式: 3（Python CLI）
2. デフォルト pytest 引数: `-q tests/perf -m perf --benchmark-only`
3. 出力先ディレクトリ: `benchmark_results/` 固定
4. 早いプリセット: 不要

## 後続アイデア（任意）

- 直近の `summary.json` と比較して差分ハイライトを HTML に表示
- 連続実行時にタイムスタンプ付きサブフォルダへ履歴保存（例: `benchmark_results/2025-09-21/`）
- `pre-commit` とは連携しない（実行時間が長いため）。必要なら別コマンドに分離。

---

本チェックリストの合意後、選択肢に応じて最小変更で実装します。
