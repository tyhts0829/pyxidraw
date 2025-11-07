# Lazy / Cache / Multiprocessing リファクタリング計画（提案）

目的
- 直近の仕様変更（LazyGeometry を関数参照ベースに刷新、キャッシュ鍵の再設計、API 注入化）後に残存している“過剰防御・重複・死蔵コード”を整理し、読みやすく・最小限で・テスト容易な実装へ磨き込む。
- 依存方向・性能・テスト安定性の維持/向上（機能追加はしない）。

非目標
- 新機能追加、キャッシュ戦略の大幅な変更、外部 API 破壊。

---

## 現状の気になる点（観測と根拠）

- engine.core.lazy_geometry
  - 環境変数ガードが過剰（try/except と Fallback の二重化）。src/engine/core/lazy_geometry.py:240
  - prefix キャッシュの署名列を計算する箇所が重複（検索時と格納時）し可読性が低い。src/engine/core/lazy_geometry.py:149, 200
  - 使われていないデバッグカウンタや関数がある可能性（_PREFIX_TAIL_SKIPPED_INC など）。src/engine/core/lazy_geometry.py:312
  - _safe_getattr/_safe_name と api.lazy_signature._impl_id で関数IDの表現が分散。src/engine/core/lazy_geometry.py:330, src/api/lazy_signature.py:18
- api/effects
  - WeakSet/compiled キャッシュの併存は必要だが、用途説明が薄く保守者負担。src/api/effects.py:27, 224
  - Pipeline.__call__ で steps を毎回コンパイル（名前→実関数）しており、Builder との責務分担が曖昧。src/api/effects.py:63
- registries（shapes/effects）
  - 既に純関数レジストリへ簡素化済みだが、docstring/型が古い説明を含む。src/effects/registry.py:1, src/shapes/registry.py:1
- 旧実装の残置
  - api/effects_old.py が未使用の可能性（参照は自ファイル内のみ）。src/api/effects_old.py:1
- renderer/runtime/export など周辺
  - OrderedDict + 環境スイッチが複数箇所で独自に実装（設定/初期化パターンの重複）。src/engine/render/renderer.py:312
  - multiprocessing 周辺は `engine/runtime/worker.py` に集約されているが、fork/child でのキャッシュ初期化について明示がない。src/engine/runtime/worker.py:18

---

## 方針（原則）
- シンプル・明快・過不足なし（不要ガード/重複の削減、早期 return、命名の明確化）。
- 共通化できる小さなヘルパは `common/` に寄せる（関数ID生成、環境値パースなど）。
- 例外は狭く・捕捉理由を明確に（裸 except を削る）。
- プロセス間共有はしない前提（現状踏襲）。プロセス生成時の初期化フックを文書化。

---

## 実施項目（チェックリスト）

Phase A: 関数ID/署名・環境値ヘルパの共通化
- [x] `common/func_id.py` 新設: `impl_id(fn) -> str`（`module:qualname` 固定、フォールバック `id(fn)`）。
  - [x] 既存 `_safe_getattr/_safe_name`（engine.core.lazy_geometry）と `api.lazy_signature._impl_id` を置換。
- [x] `common/env.py` 新設: 小さなユーティリティ（`env_int(name, default, *, min=None)` 等）。
  - [x] lazy_geometry の `_SHAPE_CACHE_MAXSIZE`/`_PREFIX_*` 初期化を置換し、try/except の重複を削減。

Phase B: LazyGeometry の簡素化と重複削減
- [x] prefix 署名列の計算を 1 箇所に集約（検索/格納の共有化）。
- [x] `_safe_getattr/_safe_name` を `common/func_id.impl_id` で置換し、ローカル実装を削除。
- [x] 未使用のデバッグカウンタ/関数の整理（lazy_geometry: 未使用カウンタ/環境フラグを削除）。
- [x] 例外捕捉の縮小とコメント整備（署名生成失敗時のみ非キャッシュにフォールバック）。

Phase C: API.effects の責務明確化
- [x] `PipelineBuilder.build()` で steps を impl 参照へ事前コンパイル（`Pipeline._compiled_steps`）。
  - [x] `__call__` は事前コンパイルがあればそれを用い、無ければフォールバックで都度コンパイル。
  - [x] キャッシュ鍵の生成は現状維持（`lazy_signature_for`）。
- [ ] WeakSet/compiled キャッシュの用途説明・カウンタ公開の整備（別途）。

Phase D: registries の doc/型整備
- [x] `effects/registry.py` と `shapes/registry.py` のモジュール冒頭 docstring を現仕様に更新（純関数登録・Lazy 非依存）。
- [ ] 型ヒントの精緻化（必要箇所を段階対応）。

Phase E: 旧実装/重複の整理
- [x] `src/api/effects_old.py` に deprecate 注記を追加（削除は後日）。
- [ ] renderer などのキャッシュ初期化パターンの重複を `common/env.py` に置換（影響小から着手）。

Phase F: multiprocessing/fork の注意書きと初期化
- [x] `engine/runtime/worker.py` に「子プロセスでのキャッシュ初期化は都度行われる（プロセスローカル）」旨を docstring に追記。
- [ ] `os.register_at_fork` で LRU をクリアするハンドラ追加（任意・未実装）。

---

## 影響範囲と互換性
- 外部 API（`G/E/shape/effect`）は非変更。
- 署名/キャッシュ鍵は現行（module:qualname + 量子化 params）から不変。
- `effects_old` は非推奨化のみ（削除しない）。

---

## 検証手順（編集ファイル優先）
- Lint/Format/Type（対象のみ）
  - `ruff check --fix {changed_files}`
  - `black {changed_files} && isort {changed_files}`
  - `mypy {changed_files}`
- テスト（対象限定）
  - `pytest -q tests/api/test_shapes_api.py -q`
  - `pytest -q tests/api/test_pipeline_cache.py -q`
  - `pytest -q tests/test_architecture.py::test_architecture_import_rules -q`
- スモーク
  - `pytest -q -m smoke`

---

## ロールバック戦略
- 各 Phase ごとに小さくコミットし、失敗時は直前の Phase まで戻せるようにする。
- 署名/キャッシュまわりはハッシュの一致を確認（差分が出た場合は原因追跡して中断）。

---

## 確認事項（要回答）
1) `common/func_id.py` / `common/env.py` の新設に同意しますか？
2) `PipelineBuilder.build()` での impl 事前コンパイル（`__call__` の簡素化）へ移行してよいですか？
3) `effects_old.py` は deprecate 表記を追加（削除は次回）で問題ありませんか？

承認後、上記チェックリストに沿って段階的に実装します。
