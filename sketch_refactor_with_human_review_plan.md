# run_sketch 完全分割計画（人間レビュー前提版）

対象: `src/api/sketch.py::run_sketch`

## ゴール

- `run_sketch` を次の 3 つのヘルパーに分割し、責務を明確にする。
  - `_setup_runtime_core(...) -> _RuntimeContext`
  - `_setup_recording_and_events(runtime: _RuntimeContext, use_midi: bool, canvas_width: int, canvas_height: int) -> None`
  - `_run_event_loop(runtime: _RuntimeContext) -> None`
- 現行挙動（UI/HUD/録画/品質モード/終了処理）を可能な限り維持する。
- 各ステップで「人間の手による動作確認」を必須とし、自動チェック（ruff/mypy）だけに頼らない。

---

## 全体方針

- 基本ポリシー:
  - 「壊さない」ことを最優先し、**小さい塊ごとに分けて → 手動確認 → 次へ**の順で進める。
  - `git diff` を細かく見る前提で、「1 コミット 1 意味のある分割」を目指す。
  - 自動チェック: `ruff` / `mypy` / `python main.py` を各ステップの最後に必ず回す。
- 観察ポイント（人間が見るべき挙動）:
  - ウィンドウ表示・ESC 終了が正常か。
  - HUD（`H` トグル）の動作。
  - PNG 保存（`P`/`Shift+P`）とエラーメッセージ。
  - G-code 開始/キャンセル（`G`/`Shift+G`）。
  - 録画開始/停止（`V`/`Shift+V`）、品質モード移行/復帰。
  - 終了時に変な例外が出ていないか（コンソール/ログ）。

---

## フェーズ 1: 下準備と安全ネット

### 1-1. 作業ブランチとスナップショット

- [ ] `git status` がクリーンであることを確認する。
- [ ] 新しいブランチを切る（例: `feat/refactor-run-sketch-split`）。
- [ ] 現行の `src/api/sketch.py` を別名保存（`sketch_backup_before_split.py` 等、コミットしないファイル）しておく。

### 1-2. ベースライン動作確認

- [ ] `python main.py` を実行し、簡単なスケッチで以下を確認:
  - [ ] ウィンドウが開く/描画される。
  - [ ] `ESC` で正常終了する。
  - [ ] `H` で HUD の表示/非表示が切り替わる。
  - [ ] `P` / `Shift+P` で PNG が保存される（パスが HUD に出る）。
  - [ ] `G` / `Shift+G` で G-code エクスポートの開始/キャンセルが動く（HUD メッセージ含む）。
  - [ ] `V` / `Shift+V` で録画開始/停止が動く（HUD の「REC」表示と品質モードメッセージ含む）。

この挙動が「正」として、以降の各フェーズで比較する。

---

## フェーズ 2: `_setup_runtime_core` の導入

### 2-1. ランタイム構成の読み取り（人間の理解ステップ）

- [ ] 現行 `run_sketch` の内部で、次のブロックをざっくりマーキングする（コメント or メモ上で良い）:
  - [ ] SwapBuffer/WorkerPool/StreamReceiver 初期化。
  - [ ] Window & ModernGL（`create_window_and_renderer` + 投影行列）。
  - [ ] HUD/MetricSampler/OverlayHUD 関連。
  - [ ] ExportService/GCodeWriter/VideoRecorder。
  - [ ] FrameClock/Tickable リストと `pyglet.clock.schedule_interval`。
  - [ ] `_RuntimeContext` への詰め込み（すでに導入済み分）。

### 2-2. `_setup_runtime_core` のスケルトン作成

- [ ] `run_sketch` の直前あたりに `_setup_runtime_core` の空スケルトンを書き、引数・戻り値を定義する:

  - 引数（最低限）:
    - `fps: int`
    - `canvas_width: int`, `canvas_height: int`, `window_width: int`, `window_height: int`
    - `background`, `line_color`, `workers`
    - `midi_manager`, `midi_service`, `cc_snapshot_fn`
    - `parameter_manager`, `draw_callable`
    - `use_parameter_gui`, `show_hud`, `hud_config`
  - 戻り値:
    - `_RuntimeContext`

- [ ] この時点では中身は「`pass`」 or `raise NotImplementedError` の状態にして、まだ呼ばない。

### 2-3. SwapBuffer/WorkerPool/StreamReceiver を `_setup_runtime_core` に移動

- [ ] 元の `run_sketch` から、以下の塊を `_setup_runtime_core` にコピー（移動）する:
  - [ ] `hud_conf = _build_hud_config(...)`
  - [ ] `swap_buffer = SwapBuffer()`
  - [ ] `_apply_cc_snapshot` の import/try-except
  - [ ] `metrics_snapshot_fn` / `_param_snapshot_fn`
  - [ ] `worker_pool = WorkerPool(...)`
  - [ ] `on_metrics_cb` / `_on_metrics` / `stream_receiver`
- [ ] `_setup_runtime_core` 内でこれらを `runtime` 用のローカル変数として束ね、最終的に `_RuntimeContext` の一部として返す。
- [ ] `run_sketch` 側では、同じ塊を削除し、`runtime = _setup_runtime_core(...)` を呼ぶだけにする。
- [ ] `ruff` / `mypy` / `python main.py` を実行して、挙動に変化がないかを確認する。

### 2-4. Window & GL / HUD / FrameClock を `_setup_runtime_core` に移動

- [ ] 次のブロックを `_setup_runtime_core` に順に移動する:
  - [ ] `proj = build_projection(...)` / `create_window_and_renderer(...)`
  - [ ] MetricSampler / OverlayHUD の初期化と設定（counts provider, extra metrics）。
  - [ ] `_apply_initial_colors(...)` / `parameter_manager.store.set_override("runner.show_hud", ...)`
  - [ ] `tickables` リストと `FrameClock` / `pyglet.clock.schedule_interval(...)`
  - [ ] `video_recorder = VideoRecorder()`
  - [ ] `_RuntimeContext(...)` 生成部分（runtime の中身）。
- [ ] `run_sketch` からは `runtime = _setup_runtime_core(...)` の戻り値だけを使い、個々のローカル変数（`swap_buffer` / `worker_pool` / `rendering_window` 等）には触らない。
- [ ] 再び `ruff` / `mypy` / `python main.py` を実行し、HUD/描画/録画まわりの挙動を手で確認する。

---

## フェーズ 3: `_setup_recording_and_events` へのイベント/録画集約

### 3-1. `_setup_recording_and_events` スケルトン作成

- [ ] `_setup_runtime_core` の下あたりに `_setup_recording_and_events(runtime, use_midi, canvas_width, canvas_height)` のスケルトンを追加。
- [ ] 引数:
  - `runtime: _RuntimeContext`
  - `use_midi: bool`
  - `canvas_width: int`, `canvas_height: int`

### 3-2. Draw callback と品質モードヘルパーを移動

- [ ] `run_sketch` 内の `_draw_main` 定義と `rendering_window.add_draw_callback(_draw_main)` を `_setup_recording_and_events` に移動し、`runtime` を使う形に書き換える。
- [ ] `_enter_quality_mode` / `_leave_quality_mode` の 2 関数も `_setup_recording_and_events` に移動し、引数ではなく `runtime` を参照するようにする。
- [ ] `run_sketch` から対応部分を削除し、代わりに `_setup_recording_and_events(runtime, use_midi, canvas_width, canvas_height)` を呼ぶ。
- [ ] `python main.py` で品質モード（録画開始/停止）の挙動を確認する。

### 3-3. キーイベント/クローズ/録画フック/シグナルを移動

- [ ] `on_key_press` / `on_close` / `_capture_frame` / シグナルハンドラ（`_sig_handler`） / `_at_exit` を `_setup_recording_and_events` に移動する。
  - [ ] いずれも `runtime` 経由でオブジェクトにアクセスするようリライトする。
  - [ ] `_shutdown_parameter_gui` も `_setup_recording_and_events` 内のローカル関数として持ち、`run_sketch` 本体からは消す。
- [ ] `run_sketch` からイベント・録画・終了処理の定義をすべて削除し、残すのは `_setup_recording_and_events(...)` 呼び出しだけにする。
- [ ] `ruff` / `mypy` / `python main.py` 実行:
  - [ ] ESC 終了、HUD トグル、PNG/G-code/録画開始停止/品質モード/終了時例外の有無を人間の目で確認。

---

## フェーズ 4: `_run_event_loop` へのループ集約

### 4-1. ループ部分の抽出

- [ ] `run_sketch` の末尾にある:
  - `_prev_excepthook = sys.excepthook`
  - `_silent_excepthook(...)`
  - `sys.excepthook = _silent_excepthook`
  - `pyglet.app.run()` / finally での excepthook 復元
  を `_run_event_loop(runtime: _RuntimeContext)` へ移動する。
- [ ] `run_sketch` の最後は `_run_event_loop(runtime)` だけにする。
- [ ] `ruff` / `mypy` / `python main.py` で、KeyboardInterrupt が従来どおり黙殺されることを確認（Ctrl+C で確認する）。

---

## フェーズ 5: 後片付け・整合性チェック

### 5-1. コード構造と依存

- [ ] `run_sketch` の行数とネストが適度なレベル（「大まかなフェーズだけ読めばよい」状態）になっているかを確認する。
- [ ] `_RuntimeContext` に余計なフィールドがないか（実際に使われていないものがないか）を確認する。
- [ ] `TYPE_CHECKING` と遅延 import の方針に従っているか確認する（`engine.*` 依存は関数内 import）。

### 5-2. テスト・動作確認

- [ ] `ruff check src/api/sketch.py`
- [ ] `mypy src/api/sketch.py`
- [ ] `python main.py` で、フェーズ 1 で列挙した観察ポイントをすべて再確認する。

### 5-3. 最終レビューとコミット

- [ ] `git diff` で `src/api/sketch.py` の変更を一通り眺め、意味のない差分（コメントの揺れなど）がないかを見る。
- [ ] 満足できる状態になっていれば、1〜数コミットにまとめて保存する（コミットメッセージ例: `refactor(api): split run_sketch setup`）。

