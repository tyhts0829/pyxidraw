# ベンチマーク設計レビュー（2025-09-04）

対象: `benchmarks/` 一式（CLI、core: config/runner/execution/types/validator/visualization、plugins: effects/shapes）。

## 現状スナップショット
- エントリポイント: `python -m benchmarks`（`benchmarks/__main__.py`）。
  - サブコマンド: run/list/validate/compare/config（テンプレ作成）。
- 設定: `benchmarks/config/default.yaml` を起点に CLI/環境変数で上書き可能（`core/config.py`）。
- 実行: `core/runner.py` がプラグインからターゲット群を収集し、逐次 or 並列で測定（ウォームアップ/ラン回数/タイムアウト）。
- 結果: `benchmark_results/` に保存（`benchmark_result_manager.py`）。`visualization.py` でグラフ/レポート生成。
- プラグイン: `plugins/effects.py`, `plugins/shapes.py`。新API（E.pipeline）準拠のパラメタライズ済みターゲットを列挙。

## 良い点
- 責務分離: CLI/Config/Runner/Plugins がほどよく分離され、拡張に強い。
- 柔軟性: run/list/validate/compare が揃い、定常運用と回帰検知の両立がしやすい。
- 既定値: ウォームアップ/ラン回数/タイムアウト/出力などの既定が妥当値で、初手から動かしやすい。
- 新 API 対応: `api.pipeline` 前提に統一（effects/shapes 両方）で設計のブレがない。

## 改善余地（観点別）
- ターゲットの粒度/タグ付け
  - 現状は `effects.*`, `shapes.*` の階層と `complexity`（shapes側）程度。対象の論理的グルーピングやタグ（e.g. `cpu-bound`, `numba`, `alloc-heavy`）があると比較に意味が出る。
- 安定性/再現性
  - ウォームアップ/測定回数はあるが、測定時の GC/スレッドプール/環境情報（CPU/NUMA/Governor 等）の簡易記録がない。比較の品質を上げるならメタを残したい。
- 結果のスキーマ/互換
  - `BenchmarkResult` はマネージャ内で保存されるが、スキーマの簡易バージョニング（`schema_version`）があると将来の互換が楽。
- 失敗時の取り扱い
  - タイムアウト/例外の集計・可視化が軽くあるが、再実行支援（failed targets のみ再測定）やスキップ・除外の指定が CLI で簡単にできると便利。
- 比較のしきい値設定
  - compare はパーセント差の閾値があるが、ターゲット/タグ単位の別閾値・最小実数差の併用など柔軟性があると実用度が上がる。
- 並列実行の粒度
  - Runner 側で `parallel=True/num_workers` はあるが、CPU 論理コア数に連動した既定、I/O バウンド向けのモード（低スレッド）など運用ガイドが欲しい。

## 推奨アクション（チェックリスト）
- [x] ターゲットタグの導入（優先: 中）
  - [x] `BenchmarkTarget` に `tags: set[str]` をオプショナルで追加（plugins がセット可能）。
  - [x] effects: `numba`/`pure-numpy`/`cache-heavy` 等、shapes: `cpu-bound`/`alloc-heavy` 等の例示実装。
  - [x] list/compare で `--tag` フィルタをサポート。
- [x] 実行メタの保存（優先: 中）
  - [x] `BenchmarkResult` に `meta: {python, os, cpu, cores, env}` などの最低限を保存（`platform`/`os`/`psutil`/`cpuinfo` 等で取得できる範囲）。
  - [x] 保存ファイルに `schema_version` を付与し、将来互換を確保。
- [x] 失敗時再実行/スキップ（優先: 低）
  - [x] run の結果から `failed_targets.json` を吐き出し、`run --from-file failed_targets.json` で再実行可能に。
  - [x] `run --skip targetA --skip targetB` 指定を CLI に追加。
- [x] しきい値柔軟化（優先: 低）
  - [x] compare の閾値に `--abs-threshold` を追加（小さな回帰/改善のノイズに対処）。
  - [x] タグ/ターゲット単位での別閾値設定（設定ファイルで上書き）。
- [x] 並列実行ガイド（優先: 低）
  - [x] README/Benchmarks ドキュメントに「CPU論理数×α」の目安や I/O バウンド時の推奨を追記。

## 実装メモ
- 互換性重視: 追加フィールドはオプショナルにして既存結果の読み込みを壊さない。
- plugins は既存の `ParametrizedBenchmarkTarget` に `tags` を任意で持たせる実装が簡単。
- メタ収集は `platform.platform()`, `sys.version`, `os.cpu_count()` 程度から段階導入。

以上。小さな PR（タグ→メタ→CLI拡張）の順で段階導入するのがおすすめです。
