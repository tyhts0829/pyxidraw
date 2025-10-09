# GLFW 65537/segfault（終了時）と `multiprocessing` セマフォ警告 — 原因と恒久対策計画

目的: ウィンドウを閉じる際に稀に出る `Glfw Error 65537: The GLFW library is not initialized` と `segmentation fault`、および `resource_tracker: leaked semaphore ...` 警告の根本原因を除去する。

## 事象（再現ログ）

- `Glfw Error 65537: The GLFW library is not initialized`
- 稀に `zsh: segmentation fault  python main.py`
- `multiprocessing/resource_tracker.py: UserWarning: leaked semaphore objects ...`
- macOS で `error messaging the mach port for IMKCFRunLoopWakeUpReliable`（実害なしの終了時警告）

## 原因分析（要約）

1) Dear PyGui の終了レース（主因）
   - Parameter GUI を Dear PyGui（DPG）で起動し、メインスレッドの `pyglet.clock.schedule_interval` から毎フレーム `render_dearpygui_frame()` を呼んでいる（src/engine/ui/parameters/dpg_window.py:116,120,426,430）。
   - 終了時に `ParameterWindow.close()` が `unschedule(self._tick)` の直後に `dpg.destroy_context()` を行うが（同:143,153）、その狭間で「最後の `_tick`」が割り込むと、GLFW 破棄後に DPG が 1 フレーム描画を試み、GLFW の C 側が 65537 を出力。稀に segfault。

2) Queue リソースのクリーンアップ不足（副次）
   - `WorkerPool.close()` で `multiprocessing.Queue.close()` は呼んでいるが `join_thread()` を呼んでいないため、終了時に `resource_tracker` がセマフォリークを警告（src/engine/runtime/worker.py:314 付近）。
   - `_WorkerProcess` が `daemon=True` のため、プロセス終了時の片付け順序で回収漏れが起こりやすい。

参考:
- Pyglet/DPG の統合方針は `architecture.md:67` や ルート `AGENTS.md:72` の「macOS ではメインスレッドから DPG を駆動」に合致している。今回は終了順序の厳密化が必要。

## 恒久対策（チェックリスト）

DPG ライフサイクル安全化（競合排除）
- [ ] `ParameterWindow` に `_closed: bool` フラグを追加。`close()` 冒頭で `True` にして以降の DPG 呼び出しを抑止。
- [ ] `_tick()` 先頭で `if self._closed: return` を追加。`dpg.is_dearpygui_running()` の前で早期 return。
- [ ] `ParameterWindow.close()` での順序を明確化: `unschedule(self._tick)` →（必要なら）`dpg.stop_dearpygui()` → `destroy_viewport()` → `destroy_context()`。
- [ ] バックグラウンドスレッド駆動経路（pyglet 不在時）は `dpg.stop_dearpygui()` → `thread.join(timeout)` を待ってから破棄。

Pyglet スケジューラ停止の明示化
- [ ] `api.sketch` の `on_close()` 先頭で `pyglet.clock.unschedule(frame_clock.tick)` を呼んでメインループ駆動を止める。
- [ ] 可能なら描画側にも `closing` ガード（任意）を設け、`draw`/HUD の残フレーム実行を抑止。

`multiprocessing` キュー/プロセスの整流化
- [ ] `WorkerPool.close()` で `mp.Queue` に対して `close()` 後に `join_thread()` を呼ぶ（`task_q`/`result_q` 双方）。`queue.Queue`（インライン時）は対象外。
- [ ] `_WorkerProcess` の `daemon=True` を見直し（`False`）。`close()` で `join(timeout)` → 生存時 `terminate()` の既存手順でハングを防止。
- [ ] 上記変更後も `leaked semaphore` 警告が出ないことを手元で確認。

ドキュメント同期
- [ ] `architecture.md` に終了順序（DPG/pyglet/WorkerPool）の要点を追記。
- [ ] ルート `AGENTS.md` の macOS 注記に「終了順序の注意」を 1 行追補（簡潔に）。

## 実施順（編集/検証の小さな単位）

1. DPG ガード追加と破棄順序の修正（`src/engine/ui/parameters/dpg_window.py`）
2. `api.sketch` の `on_close()` に `unschedule(frame_clock.tick)`（`src/api/sketch.py`）
3. `WorkerPool.close()` の join_thread と `_WorkerProcess` の `daemon` 見直し（`src/engine/runtime/worker.py`）
4. 変更ファイルに限定したチェック（以下）

## 検証（編集ファイル優先の高速ループ）

- Lint: `ruff check --fix {changed_files}`
- Format: `black {changed_files} && isort {changed_files}`
- TypeCheck: `mypy {changed_files}`
- Smoke 実行: `python main.py` → ウィンドウを開閉して 65537/segfault が出ないことを確認。
- パラメータ GUI あり/なしの双方で開閉テスト。
- `use_parameter_gui=False` でも回帰が無いことを確認。

（必要時のみ）
- テスト: `pytest -q -m smoke` あるいは関連箇所のテストを個別実行。

## リスクと緩和

- `_WorkerProcess` の `daemon=False` 化: まれに join が戻らず停止するリスク → `join(timeout)` と `terminate()` の既存フォールバックで緩和。
- DPG の API バージョン差異: `destroy_viewport()` の有無など → 例外を握り潰す try/except を維持。

## 完了条件（DoD）

- ウィンドウ閉鎖時に GLFW 65537/segfault が出ない（連続 10 回以上の開閉で再現しない）。
- `resource_tracker` の leaked semaphore 警告が消える。
- `ruff/black/isort/mypy` が変更ファイルで成功。
- 必要時、`architecture.md`/`AGENTS.md` の該当箇所が更新済み。

---

承認後、上記チェックリストに沿って実装・検証を進め、各項目をチェックしていきます。

