# run_sketch 分割計画（セットアップ関数への整理）

対象: `src/api/sketch.py::run_sketch`

## ゴール

- `run_sketch` の本体から詳細なセットアップ処理を分離し、読みやすく保守しやすい構造にする。
- 現状の挙動（API シグネチャ、UI/録画/HUD/MIDI の振る舞い）は維持しつつ、処理のまとまりごとに小さな関数へ切り出す。
- 既に導入済みの `_RuntimeContext` を活かし、状態管理・品質モード・クリーンアップ経路を明確にする。

## 分割後の構成イメージ

最終的な `run_sketch` の大まかな形:

```python
def run_sketch(...):
    fps = resolve_fps(fps)
    canvas_width, canvas_height, window_width, window_height = _resolve_canvas_and_window(...)

    midi_manager, midi_service, cc_snapshot_fn = _setup_midi_layer(use_midi)
    parameter_manager, draw_callable = _prepare_parameter_gui(...)
    if init_only:
        return

    runtime = _setup_runtime_core(
        fps=fps,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        window_width=window_width,
        window_height=window_height,
        background=background,
        line_color=line_color,
        workers=workers,
        midi_manager=midi_manager,
        midi_service=midi_service,
        cc_snapshot_fn=cc_snapshot_fn,
        parameter_manager=parameter_manager,
        draw_callable=draw_callable,
        use_parameter_gui=use_parameter_gui,
        show_hud=show_hud,
        hud_config=hud_config,
    )

    _setup_recording_and_events(runtime, use_midi=use_midi, canvas_width=canvas_width, canvas_height=canvas_height)
    _run_event_loop(runtime)
```

想定する新規ヘルパー:

- `_setup_midi_layer(use_midi) -> tuple[midi_manager, midi_service, cc_snapshot_fn]`
- `_setup_runtime_core(...) -> _RuntimeContext`
- `_setup_recording_and_events(runtime: _RuntimeContext, use_midi: bool, canvas_width: int, canvas_height: int) -> None`
- `_run_event_loop(runtime: _RuntimeContext) -> None`

既存ヘルパー（変更・統合を検討）:

- `_resolve_canvas_and_window`
- `_build_hud_config`
- `_prepare_parameter_gui`

## 詳細タスク（チェックリスト）

### 1. 構造設計

- [x] 現状の `run_sketch` 本体を「論理セクション」単位にざっくり分割し、どこまでを `_setup_runtime_core` に含めるかを決める。
- [x] `_RuntimeContext` に保持させる責務を最小限に整理し、不要なフィールド（使われていないもの）がないか確認する。
- [x] 新設する 3〜4 個のヘルパー関数のシグネチャと戻り値を確定する（上記構成案をベースに微調整）。

### 2. `_setup_midi_layer` の導入

- [x] 既存の MIDI 初期化部分（`from .sketch_runner.midi import setup_midi` ～ `midi_manager, midi_service, cc_snapshot_fn = ...`）を `_setup_midi_layer(use_midi: bool)` へ抽出する。
- [x] `run_sketch` からは `_setup_midi_layer` を呼び出し、戻り値をそのまま `runtime` 構築に渡すようにする。
- [x] 例外処理やフォールバック挙動（現状の `setup_midi` 実装依存）が変わらないことを確認する。

### 3. `_setup_runtime_core` の導入

- [ ] SwapBuffer / WorkerPool / StreamReceiver / HUDConfig / MetricSampler / OverlayHUD / ExportService / VideoRecorder / FrameClock / draw callback を初期化する処理を `_setup_runtime_core` へ移動する。
- [ ] `_setup_runtime_core` 内で `_RuntimeContext` を生成し、必要なオブジェクトをすべて詰めて返す。
- [ ] 既存のログ・例外処理（`logging.debug` / `logging.warning`）を維持しつつ、`run_sketch` 側からは `runtime` のみを扱うようにする。
- [ ] `run_sketch` 本体からは「Fps 解決」「キャンバス解決」「MIDI/GUI 設定」「init_only 早期終了」「runtime 構築」「録画/イベントセットアップ」「イベントループ開始」程度の見通しにする。

#### `_setup_runtime_core` の内部構造案

- 役割を 4 ブロックに分ける:
  - Core リソース: `SwapBuffer` / `WorkerPool` / `StreamReceiver`
  - Window & GL: `RenderWindow` / `mgl.Context` / `LineRenderer` / 投影行列生成
  - HUD & メトリクス: `HUDConfig` / `MetricSampler` / `OverlayHUD` / 追加メトリクス
  - フレーム駆動: `FrameClock` / `pyglet.clock.schedule_interval`
- 戻り値は `_RuntimeContext` のみとし、`run_sketch` 側は個々のオブジェクトに触らない。
- `use_parameter_gui` / `hud_conf.enabled` 等の条件分岐は `_setup_runtime_core` 内に閉じ込める。

### 4. `_setup_recording_and_events` の導入

- [ ] 品質モード関連のヘルパー（`_enter_quality_mode` / `_leave_quality_mode`）定義を `_setup_recording_and_events` 内に移し、`runtime` をキャプチャするローカル関数として再定義する。
- [ ] キーボードイベントハンドラ（`on_key_press`）、ウィンドウクローズハンドラ（`on_close`）、録画用描画フック（`_capture_frame`）、シグナル/atexit ハンドラ設定を `_setup_recording_and_events` に集約する。
- [ ] `on_key_press` の V キー処理・`on_close` での録画停止・品質モード復帰がすべて `_enter_quality_mode` / `_leave_quality_mode` + `runtime` を通ることを再確認する。

#### `_setup_recording_and_events` の詳細設計

- 引数: `runtime: _RuntimeContext, use_midi: bool, canvas_width: int, canvas_height: int`
- 内部で行うこと:
  - Draw callback 登録: 品質モードでの描画抑止ロジックを `runtime.quality_recording` で判定。
  - 品質モード切替ヘルパー:
    - `_enter_quality_mode()`: `api.sketch_runner.recording.enter_quality_mode` を呼び、`runtime.worker_pool` / `runtime.stream_receiver` / `runtime.frame_clock` / `runtime.quality_tick_cb` を更新。
    - `_leave_quality_mode()`: `leave_quality_mode` により上述を通常モード構成へ戻し、`runtime.quality_tick_cb` / `runtime.quality_recording` を更新。
  - イベントハンドラ:
    - `on_key_press`: ESC/PNG/G-code/HUD/V（録画）を処理。録画処理は `runtime.video_recorder` と `_enter_quality_mode` / `_leave_quality_mode` を利用。
    - `on_close`: 再入防止フラグ管理と、`FrameClock`/品質モード unschedule、キュー drain、`worker_pool.close()`、録画停止、MIDI の `save_cc()` 呼び出し、Renderer/GL 解放までを集約。
    - `_capture_frame`: 録画中のみフレーム抜き出し + HUD 描画、失敗時の録画停止・HUD 通知。
  - シグナル/atexit:
    - `_sig_handler`: `_shutdown_parameter_gui` + `runtime.rendering_window.close()`。
    - `_at_exit`: `_shutdown_parameter_gui` + `runtime.rendering_window.close()`。
  - Parameter GUI:
    - `_shutdown_parameter_gui` の実装をここに閉じ込め、`run_sketch` 本体からは触らないようにする。

### 5. `_run_event_loop` の導入

- [ ] KeyboardInterrupt 用の `sys.excepthook` 差し替え〜復元（`_silent_excepthook`）部分を `_run_event_loop(runtime)` へ抽出する。
- [ ] `run_sketch` からは `_run_event_loop(runtime)` を 1 行呼び出すだけにし、イベントループ開始〜終了処理の責務を分離する。
- [ ] 既存の挙動（KeyboardInterrupt を黙殺し、それ以外は元の `sys.excepthook` に渡す）が維持されていることを確認する。

#### `_run_event_loop` の詳細設計

- 引数: `runtime: _RuntimeContext`
- 責務:
  - `sys.excepthook` の退避と `_silent_excepthook` の設定。
  - `pyglet.app.run()` の実行。
  - finally 節での `excepthook` 復元と、復元失敗時の `logging.debug`。
- `run_sketch` からは:
  - `_setup_recording_and_events(runtime, use_midi, canvas_width, canvas_height)` 呼び出し後に `_run_event_loop(runtime)` を 1 行呼び出すだけにする。

### 6. 後片付け・整合性確認

- [ ] 分割後の `run_sketch` の行数とネストを確認し、「読む側が 1 パスでフローを追える」レベルになっているかを簡単にレビューする。
- [ ] 新設したヘルパーに対して、`TYPE_CHECKING` / import ポリシー（遅延 import）のガイドラインと矛盾がないか確認する。
- [ ] `ruff` / `mypy` / `python main.py` で再度確認し、分割による後退が無いことを検証する。

## 注意点・トレードオフ

- 分割しすぎると「どこに何があるか」を追うコストが増えるため、あくまで 3〜5 個程度の大きめヘルパーに留める。
- `_RuntimeContext` への詰め込みすぎは避け、ヘルパーのシグネチャで十分なもの（例: 一時的なローカル計算）は引数/戻り値で扱うようにする。
- 例外処理ポリシーは既に整理済みなので、分割によって再び「握りつぶし」が紛れ込まないよう、try/except ブロックは極力移動だけに留める（ロジックは変えない）。

## 実行順の提案

1. `_setup_midi_layer` の抽出（外部とのインターフェースが最も単純なため）。
2. `_setup_runtime_core` で `runtime` 構築までを一気に移動。
3. `_setup_recording_and_events` へイベント/録画ハンドラを集約。
4. `_run_event_loop` を切り出し、`run_sketch` の末尾を簡素化。
5. 全体の微調整（不要 import の削除、コメント整備、Lint/型チェック/Smoke テスト）。
