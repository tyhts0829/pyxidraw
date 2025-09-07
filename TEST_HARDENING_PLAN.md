# 追加テスト計画（Gaps & Extensions）

> 目的: 現在のユニット中心スイートで未対象／保証が弱い領域（Runner/IO/並行処理/性能/構成変更）を補完し、CLEANUP_PLAN.md の改変でも動作保証を維持する。

## 1. 不足領域と方針（要約）
- Rendering/Window/MIDI 実機: ヘッドレス前提で最小到達点までを確認（importorskip, init-only）。
- 並行処理（WorkerPool/Receiver/FrameClock）: フェイク I/O と短期タイムアウトで健全性を確認。
- 性能/予算: マイクロベンチの回帰検知を“緩い閾値”で導入（flake 回避）。
- 設定/構成変更: `configs/default.yaml` フォールバック、`src/` レイアウト移行の import 確認。
- E2E スナップショット: 代表パイプラインの digest を固定して回帰検知。
- Optional 依存の実体テスト: dummies ではなく“実依存あり”ジョブで少数ケースを実行。

## 2. マーカーと実行プロファイル
- `io`: Runner/IO/Window/MIDI 近傍の軽量到達テスト。
- `integration`: 並行処理とパイプラインの結合挙動。
- `e2e`: digest ベースの端から端までの比較（画像は対象外）。
- `perf`: 簡易ベンチ（参考値・失敗させない運用、差分レビュー用途）。
- `optional`: 実依存（shapely/fontTools/numba/mido）でのみ実行。

## 3. 構成（案）
- `tests/integration/`
  - `test_runner_init_only.py`（io, integration）
  - `test_concurrency_minimal.py`（integration）
- `tests/e2e/`
  - `test_digest_pipeline_examples.py`（e2e, snapshot）
- `tests/perf/`
  - `test_perf_smoke.py`（perf）

## 4. 具体テスト項目
- Runner/設定（io）
  - `util.utils.load_config`: `configs/default.yaml` → ルート `config.yaml` フォールバックの挙動。
  - `api.runner.run_sketch` 事前初期化到達（イベントループ開始前）。
- 並行処理（integration）
  - `WorkerPool` が `draw` を一定周期で呼び、`StreamReceiver` が `SwapBuffer` に転送する最短経路。
  - タイムアウト/停止シグナルでリークが無いこと。
- E2E digest（e2e, snapshot）
  - 代表 2〜3 ケース（例: polygon→rotate→fill, sphere→displace）を固定入力で digest 比較。
- 構成変更の耐性（integration）
  - `src/` レイアウト前提の `sys.path` 調整で import が解決できるかの検証（仮想環境下）。
- Optional 実依存（optional）
  - `effects.offset`（shapely）と `shapes.text`（fontTools）の最小ケースが ImportError なく生成/適用できる。
- 性能（perf）
  - 代表パイプラインの処理時間を計測し、過去スナップショット比で +30% 超を警告（失敗ではなくレポート）。

## 5. フィクスチャ/ユーティリティ
- `runner_headless`（io）: pyglet を importorskip、環境変数で「初期化のみ」モードを有効化。
- `fake_draw`（integration）: 低コストで `Geometry` を返す関数。
- `perf_timer`（perf）: `time.perf_counter()` の軽量ラッパ。
- 既存 `snapshot`/`digest_hex` を e2e でも流用。

## 6. 前提となる小改修（最小）
- `api.runner.run_sketch(init_only: bool=False)` を追加し、`init_only=True` で Window 生成直前に早期 return。
- `util.utils.load_config` に `configs/default.yaml` の探索を追加（無ければ従来どおり）。
- （任意）`engine.pipeline.WorkerPool` に `close()` の冪等性テスト用フック（多重 close を安全に）。

## 7. 実行コマンド（例）
- 追加領域のみ: `pytest -m "io or integration or e2e"`
- 依存ありジョブ: `pytest -m optional`
- 参考ベンチ: `pytest -m perf -q`（失敗させない）

## 8. 作業計画（テスト拡張・4人並列）
- 担当A（Runner/設定/E2E）
  - [x] `init_only` 前提テスト `tests/integration/test_runner_init_only.py`
  - [x] `load_config` フォールバックテスト（`tests/integration/test_load_config_fallback.py`）
  - [x] E2E digest 2 ケース（`tests/e2e/test_digest_pipeline_examples.py`）
  - 受け入れ基準: `pytest -m "io or e2e"` が緑。`configs/default.yaml` の値が `load_config()` に反映される。
  - 実行: `pytest -m "io or e2e"`
  - オーナーシップ: `api/runner.py`, `util/utils.py`, `tests/integration`, `tests/e2e`

### 担当A 実施ログ（2025-09-06）
- [x] `tests/integration/test_runner_init_only.py` に `@pytest.mark.io` を付与。
- [x] `tests/integration/test_load_config_fallback.py` に `@pytest.mark.integration` を付与。
- [x] 受け入れ確認実行: `pytest -m "io or e2e" -q` → 3 passed（0.48s）。
- [x] `util.utils.load_config()` が `configs/default.yaml` の `test_marker: true` を反映することを確認。
- 備考: 追加のコード改修は不要（`api.runner.run_sketch(init_only)` と `load_config()` は既に実装済）。

- 担当B（並行処理）
  - [x] `tests/integration/test_concurrency_minimal.py`（WorkerPool/Receiver/SwapBuffer 最短経路）
  - [x] 停止シグナルとタイムアウトの健全性（短時間で `close()` が多重呼び出しでも安全）
  - 受け入れ基準: `pytest -m integration` が緑。テストが 0.5s 以内に終了。
  - 実行: `pytest -m integration -q`
  - オーナーシップ: `tests/integration/*`（本体改修が必要な場合は別相談）

### 担当B 実施ログ（2025-09-06）
- [x] 追加: `tests/integration/test_concurrency_minimal.py`（2ケース）。
- [x] 最小改修: `engine/pipeline/worker.py` に `_closed` フラグを追加し `close()` を冪等化、`tick()` は閉鎖後 no-op。
- [x] 受け入れ確認実行: `pytest -m integration -q` → 4 passed（~0.9s）。
- 備考: Mac（spawn）互換のため draw はトップレベル関数＋`functools.partial` による引数バインドで実装。

 - 担当C（Optional 実依存）
  - [x] shapely/fontTools/numba/mido 実体がある環境で `-m optional` を回す小テストを追加（各1件）
  - [x] 実行手順のメモを `tests/optional/README.md` に記載（セットアップ/実行コマンド/期待）
  - 受け入れ基準: 実依存が揃った環境で `pytest -m optional` が緑（最小1件ずつ）。
  - 実行: `pytest -m optional -q`
  - オーナーシップ: `tests/optional/*`（新規）

### 担当C 実施ログ（2025-09-06）
- [x] 追加: `tests/optional/README.md`（セットアップ/実行/期待）。
- [x] 追加: `tests/optional/test_shapely_offset.py`（effects.offset の実依存確認）。
- [x] 追加: `tests/optional/test_text_fonttools.py`（shapes.Text、システムフォント探索・なければskip）。
- [x] 追加: `tests/optional/test_numba_acceleration.py`（shapes.Torus のJITパス）。
- [x] 追加: `tests/optional/test_mido_message_and_controller.py`（MIDIメッセージ処理）。
- [x] 受け入れ確認実行: `pytest -m optional -q` → 4 passed（依存あり環境）。

- 担当D（性能/構成変更）
  - [ ] `tests/perf/test_perf_smoke.py` を追加（代表1ケース、閾値超は失敗させず警告ログ）
  - [ ] `tests/integration/test_src_layout_imports.py`（仮想 `src/` レイアウトで import 確認）
  - 受け入れ基準: `pytest -m perf` が実行され、ログに実測値と比較を出力。`pytest -m integration -k src_layout` が緑。
  - 実行: `pytest -m perf -q` / `pytest -m integration -k src_layout -q`
  - オーナーシップ: `tests/perf/*`, `tests/integration/test_src_layout_imports.py`

## 9. 完了定義（Done）
- `pytest -m "io or integration or e2e"` が緑、snapshot 期待が安定。
- optional 依存の最小ケースが通る CI ジョブが存在。
- `init_only` で Runner 初期化到達のテストが確立。
- 並行処理の最短経路が短時間に安定実行、停止フローが健全。
- 代表パイプラインの digest E2E が回帰を捕捉。
