# Lazy import 簡素化計画 — 「性能だけの遅延」

- 日付: 2025-10-11（段階導入版）
- 目的: 起動高速化の利点は維持しつつ、「未導入でも動かす」ための複雑な仕掛けを撤去する。常に動作状態を保ちながら、小さなステップで安全に移行する。

## 方針（シンプル化の原則）

- 重い依存は「使う関数/メソッド内」でローカル import（性能目的の遅延）。
- トップレベルでの try-import/sentinel/専用例外は使わない（ImportError をそのまま許容）。
- 型は原則「文字列注釈」を用い、`TYPE_CHECKING` の細工は最小限。
- UI/CLI 側での“未導入検出”や案内は行わない（導入は README/エラーに委ねる）。

## 段階導入ポリシー（各フェーズ共通）

- 小さな変更単位で「変更ファイル限定」のチェックを回す。
  - Lint/Format/Type: `ruff check --fix {path}` / `black {path} && isort {path}` / `mypy {path}`
  - Test: `pytest -q -m smoke` もしくは対象テストファイルを直接指定。
- `architecture.md` と整合する説明を、該当差分が生じた段階で更新。
- 動作確認は `python main.py`（必要に応じてヘッドレス/最小パス）でスモーク。
- 破壊的変更は対象サブシステムを限定し、毎フェーズ後に“動作状態”へ復帰。

---

## フェーズ計画

### フェーズ0: 現状把握と安全網（ノンインパクト）

- 目的: 実装箇所の棚卸しと検証ループ整備のみ。挙動変更なし。
- タスク:
  - [x] 遅延/try-import/sentinel の出現箇所を `rg` で抽出し、一覧を保存。（参考: `reports/phase0_lazy_import_inventory.md`）
  - [x] 「未導入でも import 可」を前提とするテストの洗い出し（例: `tests/api/test_sketch_init_only.py`）。
  - [x] 最小スモークが通ることを確認（`pytest -q -m smoke`）。
  - [x] `architecture.md` に現状の Optional Dependencies の方針を明記（後続フェーズで更新予定の注記）。
- 完了条件:
  - [x] 出現箇所の一覧と影響テストのリスト化が完了。（`reports/phase0_lazy_import_inventory.md`）
  - [x] スモーク/変更ファイル限定チェックの運用準備が整う。

### フェーズ1: 型注釈の無害化（挙動不変）

- 目的: 実行時 import 回避のため、型参照を文字列化。`TYPE_CHECKING` の依存 import を縮小。
- タスク:
  - [x] `src/engine/render/renderer.py`, `src/util/geom3d_ops.py` などで不要な `TYPE_CHECKING` ブロックを削除、型は文字列注釈へ。
  - [x] 循環 import/実行時 import を伴う型参照を整理（必要なら forward 文字列）。
- 完了条件:
  - [x] 変更ファイルが `ruff/black/isort/mypy` を通過。
  - [x] 挙動変更がないことをスモークで確認。

### フェーズ2: moderngl（描画）をローカル import 化

- 目的: sentinel/try-import を撤去し、使用直前で `import moderngl` に統一。
- タスク:
  - [x] `src/engine/render/line_mesh.py` — トップレベル `import moderngl as mgl` を廃止（型は `Any` に置換）。
  - [x] `src/engine/render/renderer.py` — `mgl=None` sentinel/ガード撤去、`draw()` 内をローカル import に統一。
  - [x] `src/engine/export/image.py` — FBO 分岐の `importlib.import_module("moderngl")` を直接 `import moderngl` に置換。
  - [x] `src/api/sketch.py` — 現状維持（`init_only` 後のローカル import）。確認のみ。
- 完了条件:
  - [x] 変更ファイル限定のチェック合格。
  - [x] `python main.py` 最小経路のスモーク動作（代替: `pytest -q -m smoke`）。

### フェーズ3: pyglet/UI の簡素化

- 目的: sentinel/スタブ/分岐の撤去と単一ドライバ化。必要時のみローカル import。
- タスク:
  - [x] `src/engine/export/image.py` — トップレベル sentinel（`pyglet: Any = _pyglet`）を撤去し、使用スコープでローカル import。
  - [ ] `src/engine/ui/parameters/dpg_window.py` — dearpygui/pyglet の sentinel/スタブ/分岐を撤去し、直接 import＋単一ドライバへ。
  - [ ] `src/engine/core/render_window.py`, `src/engine/ui/hud/overlay.py` — 現状維持で確認。必要なら後続でローカル化検討。
- 検証:
  - [x] `pytest -q tests/ui/parameters`（パラメータ GUI の高速チェック）。
  - [ ] macOS の DPG/pyglet 駆動仕様（AGENTS.md の注記）への影響がないか確認。
- 完了条件:
  - [x] 変更ファイル限定のチェック合格＋UI 周辺のスモーク（最小経路）。

### フェーズ4: mido（MIDI）をローカル import 化

- 目的: try-import/例外ラップの撤去。必要箇所で `import mido` に一本化。
- タスク:
  - [ ] `src/engine/io/manager.py` — 関数内の try-import/例外ラップ撤去、ローカル `import mido` へ。
  - [ ] `src/engine/io/controller.py` — トップレベル `import mido` を必要箇所でのローカル import に移動。型は文字列注釈へ。
- 完了条件:
  - [ ] 変更ファイル限定のチェック合格＋該当機能の最小スモーク。

### フェーズ5: shapely（幾何エフェクト）整理

- 目的: try-import と API 差分ガードを撤去。使用関数内で直接 import に統一。
- タスク:
  - [ ] `src/effects/offset.py` — トップレベル import を削除、実処理関数内ローカル import へ。
  - [ ] `src/effects/partition.py` — try-import/API 差分ガード撤去、直接 import に整理。
- 完了条件:
  - [ ] 変更ファイル限定のチェック合格＋該当エフェクトの最小スモーク。

### フェーズ6: numba の一本化

- 目的: `_HAVE_NUMBA` 分岐/フォールバック撤去。`from numba import njit` に統一。
- タスク:
  - [ ] `src/effects/dash.py` — 分岐撤去・一本化。
  - [ ] その他（affine/displace/fill/repeat/subdivide/weave 等）— 現状維持（要確認）。
- 完了条件:
  - [ ] 変更ファイル限定のチェック合格。

### フェーズ7: フォント系のローカル化

- 目的: フォント依存（fontTools/fontPens）のトップレベル import を撤去し、需要時のみ import。
- タスク:
  - [ ] `src/shapes/text.py` — `TextRenderer.get_font()`/`get_glyph_commands()` 内にローカル import 移動。型は文字列注釈へ。
- 完了条件:
  - [ ] 変更ファイル限定のチェック合格＋フォント系スモーク。

### フェーズ8: 後片付けとドキュメント同期

- 目的: 残存ガード/スタブ/動的 import の整理と文書整合。
- タスク:
  - [ ] `reports/plan_lazy_import_alignment.md` の E/F（numba_compat と OptionalDependencyError）を撤回としてマーク（ドキュメントのみ）。
  - [ ] `architecture.md` の「Optional Dependencies」を更新（“性能目的の遅延のみ採用。未導入時は ImportError に委ねる”）。
  - [ ] importlib 動的 import（例: `src/engine/export/image.py` の moderngl 部位）を直接 import に置換（重複確認）。
- 完了条件:
  - [ ] `ruff/black/isort/mypy` 合格、関連テスト緑、スタブ同期に影響なし。

---

## 運用・テスト方針

- 影響のあるテストはフェーズ該当時にのみ更新（例: `tests/api/test_sketch_init_only.py`）。
- 各フェーズ終了時に次を実施:
  - [ ] 変更ファイル限定の Lint/Format/Type/Test。
  - [ ] 必要に応じて `pytest -q -m smoke`、UI 変更時は `pytest -q tests/ui/parameters`。
  - [ ] `python main.py` の最小スモーク。

## トレードオフ（受容）

- Pros: コード量削減、分岐/ガードの撤去で可読性向上、保守コスト低減。
- Cons: 「未導入でも壊さない」互換性を縮小。依存未導入の利用時に ImportError が発生（事前ガイドは提供しない）。

---

この段階計画に基づき、フェーズ0→1から順に最小差分で進めます。次のアクションは「フェーズ0: 棚卸し」の着手です。
