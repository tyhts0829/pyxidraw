# SwapBuffer 識別子リネーム計画（`double_buffer` → `swap_buffer`）

- 日付: 2025-10-11
- 背景: ドキュメント/コメントの用語は SwapBuffer へ統一済み。識別子（引数名/属性名/キーワード名）も合わせることで認知負荷をさらに低減する。

## 範囲と方針

- 対象: `SwapBuffer` を受け渡す変数・引数・属性などの識別子。
- 目的: `double_buffer` 系の命名を `swap_buffer` に統一。
- 除外: OpenGL のウィンドウ設定におけるダブルバッファ（`Config(double_buffer=True)`）は別概念のため除外。

## 影響箇所（インベントリ）

- engine.render
  - `src/engine/render/renderer.py:41` `def __init__(..., double_buffer: SwapBuffer, ...)`
  - `src/engine/render/renderer.py:46` `double_buffer:` の引数説明（docstring 内）
  - `src/engine/render/renderer.py:50` `self.double_buffer = double_buffer`
  - `src/engine/render/renderer.py:84` `if self.double_buffer.try_swap():`
  - `src/engine/render/renderer.py:85` `geometry = self.double_buffer.get_front()`

- engine.runtime
  - `src/engine/runtime/receiver.py:21` `def __init__(..., double_buffer: SwapBuffer, ...)`
  - `src/engine/runtime/receiver.py:32` `self._buffer = double_buffer`（内部属性名は現状 `_buffer`）

- api 層（呼び出し元）
  - `src/api/sketch.py:414` `line_renderer = LineRenderer(..., double_buffer=swap_buffer, ...)`

- 除外（非対象）
  - `src/engine/core/render_window.py:38` `Config(double_buffer=True, ...)`（OpenGL のダブルバッファ設定）

## 作業チェックリスト（Ask-first 実装前の提案）

- Renderer
  - [x] `LineRenderer.__init__` の引数名を `double_buffer` → `swap_buffer` に変更（`src/engine/render/renderer.py:41`）
  - [x] インスタンス属性を `self.double_buffer` → `self.swap_buffer` に変更（`src/engine/render/renderer.py:50, 84, 85`）
  - [x] 引数説明の文言を `swap_buffer` に更新（`src/engine/render/renderer.py:46`）

- Sketch 呼び出し元
  - [x] `LineRenderer` 生成時のキーワード引数を `double_buffer=` → `swap_buffer=` に変更（`src/api/sketch.py:414`）

- Receiver（任意: 一貫性強化）
  - [x] `StreamReceiver.__init__` の引数名を `double_buffer` → `swap_buffer` に変更（`src/engine/runtime/receiver.py:21`）
  - [x] 内部属性を `_buffer` → `_swap_buffer` に変更（`src/engine/runtime/receiver.py:32` 他参照箇所）

- 除外の確認
  - [x] `src/engine/core/render_window.py:38` の `Config(double_buffer=True, ...)` は変更しない（OpenGL 概念）

- 検証（変更ファイル限定・高速ループ）
  - [x] ruff/black/isort を変更ファイルに限定実行
  - [x] mypy を変更ファイルに限定実行（renderer/receiver に加え、呼び出し元の `sketch.py` は mypy 実行環境依存で個別実行が不安定なため、別途 CI 側で全体チェック対象）
  - [x] 影響最小テスト（`tests/api/test_shapes_api.py`）を再実行（緑）

## 後方互換についての選択肢（要確認）

- 互換キーワードの一時併用（推奨: 不要）
  - 案: `LineRenderer.__init__(..., swap_buffer: SwapBuffer, *, double_buffer: SwapBuffer | None = None)` とし、双方指定はエラーにする。
  - 現状、呼び出しは `src/api/sketch.py` のみ（社内使用）であり、互換層は不要と判断。

## リスク/影響

- 破壊的変更は `LineRenderer.__init__` のキーワード名に限定。呼び出しサイトは 1 箇所で修正容易。
- `StreamReceiver` は現状位置引数で呼び出し（`src/api/sketch.py:348`）。引数名変更の影響は限定的。

## 確認事項（質問）

1. `StreamReceiver` の内部属性 `_buffer` も `_swap_buffer` に改名しますか？（可読性向上 / 変更範囲は小）
2. `LineRenderer.__init__` に一時的な後方互換キーワード（`double_buffer`）を残しますか？（現状不要見込み）

---

本チェックリストで問題なければ実装に着手します（所要 ~10 分、変更 3 ファイル）。
