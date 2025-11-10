# ESC 終了の静粛・確実化 改善計画（実装前チェックリスト）

目的: ESC で描画ウィンドウを閉じた際に、ターミナルへのエラー出力や子プロセス残存を無くし、常に静かで確実に終了させる。

---

## 背景と現状把握（抜粋）

- フレーム駆動は `pyglet.clock.schedule_interval(frame_clock.tick, 1 / fps)` により登録
  - 参照: `src/api/sketch.py:351`
- ESC キーでウィンドウを閉じる
  - 参照: `src/api/sketch.py:453`–`src/api/sketch.py:455`
- 閉じるイベントでクリーンアップを実施（冪等フラグあり）
  - 参照: `src/api/sketch.py:513`–（`worker_pool.close()`、録画停止、MIDI 保存、`line_renderer.release()`、Parameter GUI 終了、`pyglet.app.exit()`）
- Parameter GUI（Dear PyGui）は pyglet ドライバで `schedule_interval` を用いて駆動し、終了時に `unschedule()` する
  - 参照: `src/engine/ui/parameters/dpg_window.py:1403`–`src/engine/ui/parameters/dpg_window.py:1423`
- ワーカプロセスは `multiprocessing` を使用し、`close()` で `join(timeout)` → `terminate()` まで行う
  - 参照: `src/engine/runtime/worker.py:131`（`daemon=True`）, `src/engine/runtime/worker.py:290`–（`close()`）

既知/想定される問題の源:
- ウィンドウ閉鎖後も `frame_clock.tick` が一瞬走る可能性（`unschedule` 未実施）により、GL/pyglet の解放順と競合 → 例外/ログ。
- 受信側 `StreamReceiver.tick()` が例外を親へ再送する設計のため、シャットダウン終盤のキュー残骸が端末に例外を見せる恐れ。
- `multiprocessing.Process(daemon=True)` は親終了で強制終了されるが、`join` タイミング次第で一時的な残存やエラーメッセージ発生リスク。
- macOS では UI はメインスレッド駆動が必要（AGENTS.md）。終了時の順序違反で DPG/pyglet の参照が走るとノイズになりやすい。

---

## ゴール / 非ゴール

- ゴール
  - ESC/ウィンドウクローズで「無音（stderr なし）」「確実（子プロセス/スレッド残存なし）」終了。
  - 終了コード 0。録画/エクスポート中は安全に停止/クローズ。
- 非ゴール
  - 依存追加（psutil/外部監視など）は原則行わない（Ask-first 対象）。
  - 大規模設計変更は避ける。既存構造の中で順序・保護・フックを整える。

---

## 変更方針（設計）

- シャットダウンを「二段階」に分離
  1) Quiesce 段階（ループ停止/イベント解除/例外抑止）
  2) Cleanup 段階（ワーカ停止/リソース解放/保存/アプリ退出）
- 中央集権の終了関数を導入（`graceful_shutdown()`）し、ESC/on_close/SIGINT/SIGTERM から同一路線へ集約。
- 例外は終了経路では握りつぶす（ログは DEBUG のみ）。ユーザ向け stderr は出さない。

---

## 具体的アクション（チェックリスト）

1) on_close の先頭でフレーム駆動を停止（Quiesce）
- [ ] `pyglet.clock.unschedule(frame_clock.tick)` を `on_close` の最初で実行（既に未登録なら no-op）
  - 参照: `src/api/sketch.py:351`（登録箇所）および `src/api/sketch.py:513`（on_close 開始）
- [ ] Parameter GUI ドライバは `parameter_manager.shutdown()` に委譲（現状維持）。その前にフレーム停止を済ませる。

2) 例外の抑止（Quiesce）
- [ ] `StreamReceiver` に「終了中は例外を投げない」モードを追加するか、on_close 内でキューを軽く `drain` して破棄する（小規模）
  - 低侵襲案: on_close で `while q.get_nowait(): pass` を `try/except Empty` 付きで 1 回だけ実施。
- [ ] `sys.excepthook` を `KeyboardInterrupt` 時に沈める簡易フックを `run_sketch` 実行中のみ適用（finally で復元）。

3) 子プロセス/バックグラウンドの停止（Cleanup）
- [ ] `worker_pool.close()` を呼ぶ順序を「ループ停止→（必要ならキュー破棄）→ワーカ停止」とする（現状ほぼ満たすが先頭での `unschedule` を保証）
- [ ] `worker_pool.close()` の `join(timeout)` を現状維持しつつ、タイムアウト発生時の `terminate()` 例外を握りつぶす（現状OK）。
- [ ] 録画中なら `video_recorder.stop()` を安全に実行（現状維持）。
- [ ] Parameter GUI の `shutdown()` を呼ぶ（現状維持）。

4) GL/ウィンドウ解放の順序（Cleanup）
- [ ] `line_renderer.release()` の前に描画ループ停止済みであることを保証（1) で担保）。
- [ ] ModernGL の `Context.release()` を追記（`create_window_and_renderer` の戻り `mgl_ctx` を `on_close` で明示解放）。
- [ ] 最終段で `pyglet.app.exit()`。必要なら `rendering_window.close()` の二重呼び出し防止（冪等フラグは既にあり）。

5) シグナル/終了フック（共通経路化）
- [ ] `signal(SIGINT, SIGTERM)` で `rendering_window.close()` を呼ぶハンドラを `run_sketch` 内で登録（macOS/Windows 互換の範囲）。
- [ ] `atexit` にフォールバックの `graceful_shutdown(silent=True)` を登録（GUI が既に破棄済みでも no-op）。

6) ロギング雑音の低減
- [ ] `logging.getLogger("pyglet").setLevel(logging.WARNING)` を `run_sketch` 開始時に設定（ユーザが上書き可の最小干渉）。
- [ ] 終了経路の `except Exception` は `logger.debug(..., exc_info=True)` に限定（ユーザ標準出力へは出さない）。

7) テスト/確認（スモーク）
- [ ] 手動確認（macOS/Linux）: 実行→ESC→即終了。stderr 無し、終了コード 0。
- [ ] ワーカ数 >0 と 0（インライン）でそれぞれ確認。
- [ ] DPG 有効/無効で確認。
- [ ] 録画中（V/Shift+V）に ESC で停止→保存→終了確認。
- [ ] 追加の `pytest -q -m smoke` 用に「ヘッドレス安全」スモークを検討（`init_only=True` 経路での on_close 疑似検証は別途設計）。

---

## 実装予定変更点（ファイル）

- `src/api/sketch.py`
  - `on_close` の先頭に `pyglet.clock.unschedule(frame_clock.tick)` を追加（Quiesce）。
  - `on_close` 内でのキュー `drain`（`stream_receiver.result_q` が `Queue`/`mp.Queue` のいずれでも Empty で停止）。
  - `mgl_ctx.release()` を追加（ModernGL コンテキストの明示解放）。
  - `signal`/`atexit` 設定と `graceful_shutdown()` の導入（内部関数）。
  - `sys.excepthook` の一時差し替え（KeyboardInterrupt の沈静化）。
- `src/engine/runtime/receiver.py`
  - 終了中の例外抑止モード（任意: フラグ/コンストラクタ引数）。既定は現状維持。
  - もしくは `on_close` 側の `drain` に留め、ここは変更しない（小さく済ませる）。

---

## 受け入れ基準（DoD）

- ESC/ウィンドウクローズ/CTRL+C のいずれでも終了コード 0、stderr 出力なし。
- `pgrep -fl python` 等で子プロセスが残らない（手動確認）。
- `ruff/mypy/pytest -q -m smoke` 緑（変更ファイル限定で確認）。
- 依存追加なし、Ask-first 操作なし。

---

## リスクと緩和

- ループ停止のタイミング変更により録画フレーム取りこぼし: 終了時のみであり許容。録画中は `stop()` を最優先で呼ぶ順序を維持。
- `daemon=True` のままでも `join→terminate` を行うため、異常残存は低確率。必要時に `daemon=False` 検討（今回の範囲外）。
- GL リソースの解放順序を誤ると Assertion/ログが出る可能性: 先に `unschedule` 済みで回避。

---

## 確認事項（ご相談）

- 例外の完全サプレッション（終了経路）は許容しますか？ DEBUG ログのみに限定します。
- `StreamReceiver` にフラグ追加（終了中は例外を投げない）を行うか、on_close の `drain` だけで済ませるか、どちらが好みですか？
- ワーカ `Process` の `daemon=True` は現状維持でよいですか？（`daemon=False` へ変更すると親終了待ちの設計に寄る）
- `pyglet` ロガーの既定レベルを WARNING へ引き上げることに問題はありませんか？
- `SIGINT/SIGTERM` を `rendering_window.close()` へ束ねる実装でよいですか？

---

## 実施順序（タスク分割）

- [ ] A. `unschedule` 追加と on_close の順序見直し（最小差分）
- [ ] B. ModernGL コンテキスト `release()` の追加
- [ ] C. 受信キューの `drain` 追加 or `StreamReceiver` の終了フラグ実装（どちらか一方）
- [ ] D. signal/atexit/KeyboardInterrupt フックの追加
- [ ] E. ログ・レベル調整（pyglet のみ）
- [ ] F. 手動スモーク（DPG 有無/録画有無/ワーカ数）

---

メモ: コード変更はこの計画の確認後に着手します。必要な追加の観点や方針変更があれば、この md に追記して合意を取った上で進めます。

