# ESC 終了時の「Glfw Error 65537」/ segfault 調査メモ

目的: ESC で描画ウィンドウを閉じた直後に発生する `Glfw Error 65537: The GLFW library is not initialized` と segfault の原因を、実装読みから特定し、対策候補を整理する。

## 事象要約

- 再現: `python sketch/251113.py` を実行し、ESC で終了。
- 直後に以下が出力される。
  - `Glfw Error 65537: The GLFW library is not initialized`
  - `segmentation fault`
- pyglet 側ではなく、Dear PyGui(GLFW) 側で未初期化エラーが報告された後にプロセスが落ちる。

## 関連実装

- ESC ハンドラ: `src/api/sketch.py:486` の `on_key_press()` で `rendering_window.close()` を実行。
- 終了処理（on_close）: `src/api/sketch.py:546` の `on_close()` が冪等に全クリーンアップ。
  - `pyglet.clock.unschedule(frame_clock.tick)` 等で自前スケジューラを停止。
  - レンダラ/ModernGL/MIDI を解放。
  - 最後に `pyglet.app.exit()`。
- Parameter GUI(Dear PyGui) の駆動:
  - `src/engine/ui/parameters/dpg_window.py:1508` 付近
    - macOS では主スレッドの `pyglet.clock.schedule_interval(self._tick, 1/60)` で `dpg.render_dearpygui_frame()` を呼び続ける設計。
  - 停止: `src/engine/ui/parameters/dpg_window.py:1618` の `_stop_driver()` が `pyglet.clock.unschedule(self._tick)` を実行。
  - クローズ: `src/engine/ui/parameters/dpg_window.py:115` の `close()` が `hide_viewport() → destroy_context()` を呼ぶ。
- Parameter GUI の寿命管理:
  - マネージャ: `src/engine/ui/parameters/manager.py:253` の `shutdown()` が `save_overrides(self.store)` → `controller.shutdown()`（= `window.close()`）の順で実行。
  - ストア購読: `src/engine/ui/parameters/dpg_window.py:93` で `self._store.subscribe(_on_store_change_wrapper)`（解除は未実装）。

## 時系列（推定）

以下は ESC 押下時の主な出来事の順を、コード参照付きで示す（正確な順序は pyglet のイベントループ実装に依存しうるが、問題の再現に十分な近似）。

1) `on_key_press(ESC)` が呼ばれ、`rendering_window.close()` を実行（`src/api/sketch.py:486-489`）。
2) pyglet が `on_close()` をディスパッチ（`src/api/sketch.py:546-613`）。
3) `on_close()` の冒頭で Parameter GUI のシャットダウンが走る（`parameter_manager.shutdown()`）。
   - `save_overrides()` により `ParameterStore._notify()` が発火し、DPG 購読者 `_on_store_change_wrapper()` が走る（`src/engine/ui/parameters/persistence.py` → `state.py:198` → `dpg_window.py:90-97`）。
   - 直後に `controller.shutdown()` → `ParameterWindow.close()` が呼ばれ、
     - `_stop_driver()` で `pyglet.clock.unschedule(self._tick)`（`dpg_window.py:1618-1620`）。
     - `dpg.hide_viewport()` → `dpg.destroy_context()`（`dpg_window.py:115-126`）。
4) ここでレースが発生: `unschedule(self._tick)` 実行“前後”のどこかで、同フレーム内に既にキューされた `self._tick` が一度だけ実行される可能性がある。
   - `self._tick()` は無条件で `dpg.render_dearpygui_frame()` を呼ぶ（`dpg_window.py:1510-1516`）。
   - そのタイミングで DPG コンテキストが既に `destroy_context()` されていると、GLFW 未初期化の呼び出しとなり、`Glfw Error 65537` → segfault に至る。

補足: DPG の購読解除が無い（`unsubscribe()` を未使用）ため、`ParameterWindow` 破棄後も `ParameterStore._notify()` により `_on_store_change_wrapper()` が呼ばれ得る。`_on_store_change_wrapper()` は内部で `sync_display_from_store()` を呼び、多数の `dpg.*` API を叩くため、コンテキスト破棄後に到達すると同様のクラッシュを誘発しうる。

## 根本原因（まとめ）

- Dear PyGui のフレーム駆動（`pyglet.clock.schedule_interval(self._tick, ...)`）が、
  - コンテキスト破棄直前/直後の“最後の1回”を実行し、
  - その中で `dpg.render_dearpygui_frame()` が GLFW 呼び出しを行う。
- または、`ParameterStore` の購読通知が、`destroy_context()` 後に `dpg.*` API を呼ぶ経路（購読解除漏れ）。
- いずれも「GLFW 終了後に GLFW API が呼ばれる」ことが本質で、`65537` → segfault に繋がる。

## 実装から見える脆弱点

- フレームドライバの停止とコンテキスト破棄の間にレース窓がある。
  - `_stop_driver()` は unschedule のみで「実行済み/直前にキューされた1回」を防げない。
- `ParameterWindow` は `ParameterStore` から購読解除していない。
  - 破棄後に `_on_store_change_wrapper()`→`sync_display_from_store()` が走る可能性が残る。
- `self._tick()` に「閉鎖フラグ」「dearpygui 実行可否チェック」が無い。
  - `dpg.is_dearpygui_running()` や `dpg.is_viewport_ok()` の確認が無い。
- 終了フロー上に「非同期の確実なバリア」が無い。
  - 例: unschedule 直後に `pyglet.clock.schedule_once(..., 0.0)` で destroy を次フレームに送る等の工夫が未実装。

## 対策候補（実装変更案、まだ適用しない）

優先度順に整理。全てフェイルソフトで、実装は簡潔に保つ。

1) ParameterWindow の `_tick()` にガードを追加
   - `_closing` フラグ（`close()` で True）を導入し、True なら即 return。
   - 併せて `dpg.is_dearpygui_running()` と `dpg.is_viewport_ok()` をチェックし、偽なら return。

2) `ParameterWindow.close()` の改善
   - `_closing = True` を最初に設定。
   - `_stop_driver()`（unschedule）→ `dpg.hide_viewport()` → `dpg.destroy_context()` の順を維持。
   - `ParameterStore.unsubscribe(_on_store_change_wrapper)` を追加して購読解除。

3) `on_close()` の順序調整（同期バリアの導入）
   - `parameter_manager.shutdown()` を「unschedule だけ先に行い、`destroy_context()` は `pyglet.clock.schedule_once(..., 0.0)` に寄せる」か、
     あるいは `pyglet.app.exit()` 後の `atexit` ハンドラで DPG を破棄する。
   - 目的は「現在フレームの残り tick を確実に枯らしてから destroy」すること。

4) 購読解除の徹底
   - `ParameterWindow` 側で購読解除を実施し、破棄後の `dpg.*` 呼び出し経路を閉じる。

5) 追加の安全策
   - `_tick()` の `render_dearpygui_frame()` を `try/except` で保護（ただし segfault は Python 例外で捕まらない場合がある点に留意）。

## 影響範囲とリスク

- 1)〜2) はローカル変更で副作用が小さく、まず適用候補。
- 3) はイベント順序に影響するため要注意。UI の残フレームでの挙動（HUD 反映の遅延など）に軽微な影響があり得る。
- 4) は購読の生存期間を明確化し、将来の類似クラッシュを予防。

## 次アクション（提案チェックリスト）

- [x] `_tick()` に `_closing` と `is_dearpygui_running()/is_viewport_ok()` ガードを導入。
- [x] `ParameterWindow.close()` で `_closing=True` 設定と `unsubscribe()` を追加。
- [ ] `on_close()` で DPG 破棄を「unschedule → 次フレームで destroy」に変更するか検討（実験）。
- [ ] 再現確認（ESC 終了で `65537`/segfault が消えることを確認）。
- [ ] ドキュメント更新（本メモを要約して `architecture.md` か AGENTS に既知事項として追記）。

---

補記: ModernGL/pyglet/録画（VideoRecorder）経路は GLFW を用いないため今回の直接因ではない。`src/api/sketch_runner/params.py` の購読者は pyglet 経由の apply であり DPG 非依存。問題は DPG(GLFW) のライフサイクルとフレーム駆動の競合に限られる。

## Multiprocessing 影響調査（関係の有無）

本リポは描画用データ生成を `multiprocessing` で並列化していますが、今回の GLFW 未初期化エラーとの因果関係は現状薄いと判断します。実装根拠と考察を示します。

関連実装と責務分離
- Worker プロセスの生成箇所: `src/engine/runtime/worker.py`
  - `WorkerPool(num_workers>0)` が `mp.Process` を起動（`_WorkerProcess`、`run()` は CPU 側の `draw_callback` のみ実行）。
  - GL/pyglet/DearPyGui を import/初期化しない設計。戻り値は `RenderPacket` をキューで親へ返すのみ。
- 受信側: `src/engine/runtime/receiver.py`
  - 結果キューから `RenderPacket` を取り出すだけ。UI や DPG API を触らない。
- Parameter GUI の値をワーカへ伝える経路: `src/engine/ui/parameters/snapshot.py`
  - `extract_overrides()` で値を辞書化して渡すのみで、Dear PyGui を使わない。

終了フローでのワーカ停止
- `on_close()` で `worker_pool.close()` を呼び、`None` センチネル送信→`join()`→必要時 `terminate()` を実施（`worker.py:257-311`）。
- `StreamReceiver` のキューは drain 済み（`src/api/sketch.py:569-582`）。
- これらは GLFW/DearPyGui に未接触。

考えられる間接影響と評価
- イベントスケジューリングの相互作用: ワーカ/受信/レンダラは `FrameClock` に束ねられ、`pyglet.clock.schedule_interval(frame_clock.tick, ...)` で駆動。DPG は別に `schedule_interval(self._tick, ...)`。終了時の unschedule 順序により「最後の1回」のタイミングは変動し得るが、問題の本質（GLFW 終了後の DPG 呼び出し）は解消しない。
- 品質モード切替: `src/api/sketch_runner/recording.py` では一時的に inline ワーカ（`num_workers=0`）へ切替。終了ハンドラでは無条件で `_leave_quality_mode()` を呼ぶため、終了直前に通常モードの `WorkerPool` を再作成・再スケジュールする経路がある（`src/api/sketch.py:587-590` 付近）。これは終了手順としては冗長で、将来的に整理すべきだが、GLFW 未初期化の直接原因ではない（DPG を触らない）。
- macOS の spawn 方式: 子プロセスは独立したアドレス空間を持ち、親の GLFW 状態に干渉しない。`set_start_method()` の上書きも本リポでは行っていない。

切り分けの実験提案（再現確認）
- `sketch/251113.py:14` の `workers=4` を `workers=0`（インライン実行）に変更し、ESC 終了で現象が残るか確認。
- 予想: 現象は残存する（DPG 側のフレーム駆動レースが原因のため）。

結論
- multiprocessing 自体は本件の直接原因ではない。DPG(GLFW) のライフサイクルとフレーム駆動のレースが主因。
- とはいえ、終了時に `_leave_quality_mode()` が無条件で通常モードへ復帰して再スケジュールする点は、終了経路を複雑化するため、別途是正候補。
