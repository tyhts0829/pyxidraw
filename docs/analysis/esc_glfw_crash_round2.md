# ESC 終了時の GLFW 65537 再発（2nd 分析）

現象: 最小化後の再実行で、以下のログとともに再びクラッシュ。

```
error messaging the mach port for IMKCFRunLoopWakeUpReliable
'created' timestamp seems very low; regarding as unix timestamp
1 extra bytes in post.stringData array
Glfw Error 65537: The GLFW library is not initialized
segmentation fault
```

## まとめ（暫定結論）

- 先の修正で追加した「Dear PyGui 側のガード（_closing, is_dearpygui_running/is_viewport_ok）」と「ParameterStore 購読解除」だけでは、macOS の終了タイミング依存のレースを完全には防げない。
- `on_close()` で Dear PyGui（Parameter GUI）を“最初に停止”しておく順序が、実運用上は必要。
  - これを外し（停止を末尾に戻し）たことで、再び 65537 → segfault が発生したと考えられる。
- 追加で観測された IMK/CFRunLoop のログは macOS の入力方式（IME）・ランループ絡みのノイズで、GLFW 65537 の直接原因ではないが、「終了時のイベントがまだ飛んでいる」ことを示唆する。

結論: エラー解消には「DPG の早期停止 + ガード/購読解除」の併用が最小構成。

## 根拠（コードと時系列）

参照:
- ESC キー: `src/api/sketch.py:486` の `on_key_press()` で `rendering_window.close()`
- 終了ハンドラ: `src/api/sketch.py:546` の `on_close()`
- Dear PyGui 駆動: `src/engine/ui/parameters/dpg_window.py:1568-1576`（`pyglet.clock.schedule_interval(self._tick, ...)`）
- DPG 終了: `src/engine/ui/parameters/dpg_window.py:115-126`（`_closing=True` → `unschedule` → `hide_viewport` → `destroy_context`）

変更の影響:
- 第1回修正では、`on_close()` の冒頭で `parameter_manager.shutdown()` を呼び、DPG をいち早く停止。
  - この状態では 65537/segfault は発生しなかった（ユーザー報告）。
- 最小化のため、DPG 停止を `on_close()` の末尾へ戻したところ、再発。

時系列（再発パターンの推測）:
1) `on_close()` 進行中、DPG はまだスケジュールされた `_tick()` を実行可能。
2) macOS の IME/ランループ由来のイベント（IMKCFRunLoop...）が届くなど、終了直前に DPG フレームが実行。
3) `on_close()` の末尾で `parameter_manager.shutdown()` が DPG を破棄（`destroy_context()` → 内部で `glfwTerminate()`）。
4) 破棄直後、何らかの残タスク/イベントが DPG/GLFW の関数を呼び、`GLFW_NOT_INITIALIZED(65537)` を誘発 → segfault。

備考:
- `_tick()` には `_closing` ガードを入れてあるが、`_closing=True` が立つのは `close()` 呼び出し時。DPG を「末尾で破棄」する構成だと、破棄直前まで `_closing` が False のままで最後の1回が走りうる。
- `unsubscribe` 済みでも、OS 側イベント（IME 等）は購読経路外で走る可能性があるため、ガードだけでは完全に抑止できない。

## なぜ早期停止が効くのか

- DPG（GLFW）を最初に停止しておけば、その後の OS イベントや pyglet スケジューラ内の残処理から DPG/GLFW に達しない。
- これにより、`destroy_context()`（= `glfwTerminate()`）後に DPG/GLFW API が呼ばれる機会を実質的にゼロにできる。

## 代替仮説の検討

- スレッドドライバ経路（バックグラウンドスレッド）による不具合の可能性: 本実装は `pyglet` が存在する限りメインスレッドのスケジューラを用いる（`dpg_window.py:1568-1576`）。今回の現象でも pyglet は利用されており、スレッドドライバは使っていない想定。
- ModernGL や pyglet 側の GL teardown との干渉: DPG は独立に GLFW/GL を扱うため直接干渉は薄い。ただし終了タイミングが近いと OS/ドライバのイベント駆動でレースが顕在化しやすい。

## 推奨（必要最低限の戻し）

- `on_close()` の最初に `parameter_manager.shutdown()` を戻す（DPG を先に停止）。
- 既に実装済みの以下はそのまま維持（安全性に寄与、複雑化は最小）:
  - `_tick()` の `_closing` ガード + `is_dearpygui_running()/is_viewport_ok()` チェック。
  - ParameterStore 購読解除（`unsubscribe`）。
  - `hide_viewport()` → `destroy_context()` の順序。

## 追加の計測/検証案（実装外）

- DPG ドライバ種別のログ出力（pyglet or thread）を INFO で出す。
- `on_close()` 内で DPG 停止直後に `pyglet.clock.tick()` を1回挟んで、残キュー消化後に他の teardown を進める（要検証）。

---

この分析は「コード変更なし」の調査結果です。実際の修正は上記「推奨」を反映することで再発を抑止できます。必要なら、最小差分で適用します。

