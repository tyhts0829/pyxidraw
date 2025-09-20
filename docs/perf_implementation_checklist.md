# ベンチ実装チェックリスト（pytest-benchmark＋メモリ計測）

本チェックリストは、依存追加とテスト整備を伴う実装作業を段階ごとのチェックリストで管理する。依存追加は Ask-first（承認後に着手）。

## 進捗サマリ（Stages）
- [x] Stage 0: 仕様合意（ドキュメント整備）
- [ ] Stage 1: 依存追加（dev）【Ask-first】
- [ ] Stage 2: ユーティリティ＋最小ベンチ（Geometry）
- [ ] Stage 3: Pipeline ベンチ
- [ ] Stage 4: Renderer 前処理ベンチ
- [ ] Stage 5: ベースライン保存と比較手順整備
- [ ] Stage 6: CI 連携（別PR・Ask-first）
- [ ] Stage 7: 回帰ゲート（オプトイン）

## Stage 0: 仕様合意（このドキュメント）
- [x] `docs/benchmarks.md` を pytest-benchmark＋メモリ計測前提に更新
- [x] 本チェックリスト（docs/perf_implementation_checklist.md）を追加
- 完了指標: ユーザー承認（依存追加とテスト方針への合意）

## Stage 1: 依存追加（dev）【Ask-first】
- [ ] `pyproject.toml` dev へ `pytest-benchmark>=4.0.0` を追加
- [ ] `pyproject.toml` dev へ `psutil>=5.9` を追加
- [ ] ローカル導入: `python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`
- [ ] 動作確認: `pytest --help` に benchmark オプション表示
- [ ] 動作確認: `python -c "import psutil"` 成功
- 完了指標: すべての動作確認 OK

## Stage 2: ユーティリティ＋最小ベンチ（Geometry）
- [ ] `tests/perf/_mem.py` に `measure_memory(fn, *args, **kwargs)` を実装
- [ ] `tests/perf/test_geometry_perf.py` に `Geometry.from_lines`（S/M）を追加
- [ ] `tests/perf/test_geometry_perf.py` に `rotate/scale/concat` Micro を追加
- [ ] 乱数固定（必要時）と `benchmark.extra_info` へ `N/M`＋メモリ指標を付与
- 完了指標: `pytest -q -m perf --benchmark-only tests/perf/test_geometry_perf.py` 緑（extra_info 出力確認）

## Stage 3: Pipeline ベンチ
- [ ] `tests/perf/test_pipeline_perf.py` に miss/hit 計測
- [ ] digest on/off 計測（`PXD_DISABLE_GEOMETRY_DIGEST`）
- [ ] cache on/off 計測（`E.pipeline.cache(maxsize=...)`）
- 完了指標: `pytest -q -m perf --benchmark-only tests/perf/test_pipeline_perf.py` 緑（miss/hit 差分の可視化）

## Stage 4: Renderer 前処理ベンチ
- [ ] `tests/perf/test_renderer_utils_perf.py` に `_geometry_to_vertices_indices`（S/M）
- 完了指標: テスト緑、`ops/rounds` と extra_info が付与される

## Stage 5: ベースライン保存と比較手順整備
- [ ] autosave 実行: `pytest -q -m perf --benchmark-only --benchmark-autosave`
- [ ] 比較実行: `pytest -q -m perf --benchmark-only --benchmark-compare`
- [ ] JSON 出力: `pytest -q -m perf --benchmark-only --benchmark-json=perf.json`
- [ ] `docs/benchmarks.md` にコマンド例と出力位置を追記（必要なら）
- 完了指標: `.benchmarks/` 生成・比較・`perf.json` 出力を確認

## Stage 6: CI 連携（別PR・Ask-first）
- [ ] perf ジョブ追加（`pytest -q -m perf --benchmark-only --benchmark-autosave --benchmark-json=perf.json`）
- [ ] `.benchmarks/` と `perf.json` をアーティファクト化
- 完了指標: CI 上で perf 実行とアーティファクト保存を確認

## Stage 7: 回帰ゲート（オプトイン）
- [ ] 閾値方針を決定（例: 直近中央値比 +20%）
- [ ] 環境変数でゲート有効化（例: `PXD_PERF_ENFORCE=1`）と判定実装
- [ ] ローカルで意図的負荷により失敗を再現（正常系通過を確認）
- 完了指標: 有効化時に回帰検知で失敗、無効時はスモークとして通過

---

補足（ノイズ低減・再現性）
- 推奨環境変数: `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- ランダムは固定シード（`np.random.default_rng(0)`）
