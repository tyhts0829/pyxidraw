# 遅延 import 方針 — 揃え込みチェックリスト（optional 依存の統一）

- 日付: 2025-10-11
- 目的: 「オプショナル依存はモジュール import を壊さない」を徹底するため、try-import／遅延 import／TYPE_CHECKING を統一適用。
- 対象依存: moderngl, pyglet, mido（拡張: numba, shapely）
- 実装パターン（方針）
  - 先頭で try-import → 失敗時は sentinel(None) を代入し、使用時にガード。
  - 社内重モジュールはローカル import（関数/メソッド内）。
  - 型は `from __future__ import annotations` と `TYPE_CHECKING` を使用。
  - 実行時に依存が必要な箇所は OptionalDependencyError を投げる（util.optional の追加を検討）。

## A. moderngl 揃え込み
- [ ] src/engine/render/line_mesh.py
  - 先頭の `import moderngl as mgl` を `try: import moderngl as mgl; except: mgl=None` に変更。
  - 型注釈は文字列参照（"mgl.Context"）に維持。実行時参照は ctx のみを使用。
  - 参考: src/engine/render/renderer.py のパターン。

## B. pyglet 揃え込み
- [ ] src/engine/core/render_window.py
  - 先頭の `import pyglet` / `from pyglet.gl import ...` を try-import に変更。
  - `RenderWindow` を pyglet 未導入時も定義可能に（インスタンス化で OptionalDependencyError を明示）。
  - `glClearColor` の参照は使用直前に取得 or ガード。
- [ ] src/engine/ui/hud/overlay.py
  - 先頭の `import pyglet` / `from pyglet.shapes import Rectangle` / `from pyglet.window import Window` を try-import に変更。
  - TYPE_CHECKING 下でのみ `Window` 型を import。Rectangle は使用時ガード。
  - pyglet 未導入時は No-op の描画（もしくはコンストラクタで OptionalDependencyError）。
- [ ] src/api/sketch.py（確認）
  - 既に遅延 import（moderngl/pyglet）済み。変更不要か確認のみ。
- [ ] src/engine/ui/parameters/dpg_window.py（確認）
  - 既に sentinel 運用済み。変更不要か確認のみ。

## C. mido（MIDI）揃え込み
- [ ] src/engine/io/controller.py
  - 先頭の `import mido` を try-import に置換（sentinel）。
  - mido 未導入時でもモジュール import が成功するようにし、実行時に OptionalDependencyError を送出。
  - manager 側の遅延 import と整合（モジュール import で落ちない）。
- [ ] src/engine/io/manager.py（確認）
  - 遅延 import 実装済み。エラーメッセージ体裁を OptionalDependencyError に統一（任意）。

## D. shapely（Effects）揃え込み
- [ ] src/effects/offset.py
  - 先頭の `from shapely...` を関数内遅延 import に変更（`_buffer` 内で import）。
  - 未導入時は OptionalDependencyError を送出（実行時）。
- [ ] src/effects/partition.py
  - 先頭/関数先頭の shapely import を try-import に整理し、未導入時は OptionalDependencyError。

## E. numba（njit）互換レイヤ追加と置換
- [ ] 新規: `src/util/numba_compat.py`
  - 実装: `try: from numba import njit, types; except: 定義互換のダミー njit/types を提供`。
  - njit ダミーはデコレータ透過（元関数を返す）。
- [ ] 置換: `from numba import njit` を `from util.numba_compat import njit` へ変更
  - 対象（検索一致）:
    - src/util/geom3d_ops.py
    - src/util/polygon_grouping.py
    - src/effects/affine.py
    - src/effects/collapse.py（既に一部フォールバックあり・統一）
    - src/effects/dash.py
    - src/effects/displace.py
    - src/effects/fill.py
    - src/effects/repeat.py
    - src/effects/subdivide.py
    - src/effects/weave.py
    - src/shapes/asemic_glyph.py
    - src/shapes/capsule.py
    - src/shapes/text.py
    - src/shapes/torus.py
  - `from numba import types` を参照している箇所は `from util.numba_compat import types` に切替。

## F. 例外/ヘルパ（任意だが推奨）
- [ ] 新規: `src/util/optional.py`
  - `class OptionalDependencyError(RuntimeError)` を定義。
  - `def require(module_name: str, feature: str, hint: str) -> NoReturn` で整ったエラーメッセージを生成。
- [ ] 上記モジュールで未導入時の例外を OptionalDependencyError に統一。

## G. 検証（変更ファイル限定の高速ループ）
- [ ] 変更ファイルに対して ruff/black/isort 実行
- [ ] 変更ファイルに対して mypy 実行
- [ ] 影響最小テストを実行
  - [ ] `pytest -q tests/api/test_shapes_api.py`
  - [ ] `pytest -q tests/api/test_sketch_init_only.py`（init_only=True で重依存を import しないことを確認）
  - [ ] `pytest -q -m optional`（環境に応じて）

## H. ドキュメント
- [ ] architecture.md に「Optional Dependencies & Lazy Import」節を追加（方針/パターン/例）。

---

補足
- 変更は import 安定性の向上が目的。実行時に依存が必要な箇所では、既存の UI 挙動（No-op/スキップ）か明示エラーの方針をファイルごとに選択。
- 互換レイヤ（numba_compat）は optional 依存が無い環境でも import に失敗しない土台として利用。
