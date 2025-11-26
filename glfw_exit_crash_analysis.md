# ESC 終了時の GLFW セグフォ調査メモ

## 症状
- `python main.py` 実行後に ESC で終了すると `Glfw Error 65537: The GLFW library is not initialized` が出て Segmentation fault。
- 親プロセスが即死するため WorkerPool のサブプロセスが残存。

## 調査ログ
- クラッシュレポート `~/Library/Logs/DiagnosticReports/python3.10-2025-11-26-174835.ips` / `...-174442.ips` を確認。
- main thread スタックは `mvRenderFrame() -> render_dearpygui_frame`（Dear PyGui/GLFW）呼び出し中に EXC_BAD_ACCESS。呼び出し元は CFRunLoop の Timer（pyglet clock が使うタイマー経由）。
- つまり終了シーケンス中も Dear PyGui の描画コールバックが走り、破棄済みの GLFW コンテキストに触れて落ちている。

## 原因推定
- Parameter GUI（Dear PyGui）のドライバが pyglet clock に残ったまま、ESC 終了で GLFW が破棄された後に `render_dearpygui_frame` が 1 フレーム動き、`GLFW library is not initialized` → セグフォになっている。
- セグフォで `on_close` 後半（WorkerPool.close など）が未実行になり、サブプロセスが孤児化。

## 対策案
- Shutdown の徹底: `on_close` 先頭で ParameterWindow のドライバ停止と `dpg.stop_dearpygui()`、`_closing` フラグ設定を必ず実行し、unschedule 失敗時はログに残す。
- 再入防止: ParameterWindow._tick で `_closing` または `not dpg.is_viewport_ok()` / `not dpg.is_dearpygui_running()` を検出したら即 return し、必要ならここでも unschedule を再実行して二重ガードを掛ける（GLFW 未初期化時に render を呼ばない）。
- フェイルセーフ: atexit/signal でも ParameterWindow.close を呼ぶようにして、pyglet 側の終了順序に依存せず GLFW を先に止める。
- 回避策: 修正まで `run(..., use_parameter_gui=False)` で Parameter GUI を無効化すれば GLFW を触らないためクラッシュは回避できる。
