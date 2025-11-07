# 依存関係リファクタ計画（ゼロベース設計）

目的: engine.core（L0）から registries（shapes/effects, L1）への依存と循環を完全排除し、レイヤ/禁止エッジ/循環テストを全緑にする。性能・API 体験（Lazy 既定）は維持しつつ、構造をシンプルにする。

スコープ: コード本体の構造変更・一部 API の内部仕様変更（外部 API は維持）。テスト・ドキュメント更新を含む。

非目標: 新機能の追加、描画アルゴリズムの変更。

---

## 現状の問題（要因）

- engine.core.lazy_geometry から registries 参照（禁止エッジ/レイヤ違反/循環）
  - realize 内の名前解決で `shapes.registry`/`effects.registry` を import。
  - plan も効果名ベースのため、適用時に registry を参照。
- registry が LazyGeometry を返すラッパを公開（ドメイン層が core 型に依存）
  - 設計上は許容（L1→L0）だが、core 側が逆参照することで循環が顕在化。

---

## あるべき姿（依存の方向）

- L0（engine.core, common, util）: どこからも参照可能だが、上位（L1/L2/L3）を参照しない。
- L1（effects, shapes）: L0 には依存可。registries は「純粋な関数レジストリ」に徹する。
- L2（engine.render/pipeline/ui/io/monitor）: L0 に依存、L1 直接参照は原則避ける。
- L3（api）: 唯一の registries 参照地点。ユーザー向けの組み立て（G/E/pipeline）。

キーポイント
- engine.core は「関数ポインタ」を受け取り実行するだけ（名前解決しない）。
- registries は LazyGeometry/Geometry を知らなくても良い（純関数の登録/取得）。
- API 層で名前→実関数（orig impl）に解決し、Lazy spec に関数参照を注入。

---

## 提案する構造（ゼロベース）

1) LazyGeometry の仕様変更（データ設計）
- base_kind: `"shape" | "geometry"`（現状維持）
- base_payload（shape の場合）: `(impl: Callable[..., Geometry|LineLike], params: dict)` に変更。
- plan: `list[tuple[impl: Callable[[Geometry], Geometry], params: dict]]` に変更。
- realize():
  - base が geometry ならそのまま。
  - base が shape なら `impl(**params)` を実行（lines は `Geometry.from_lines` で正規化）。
  - plan は `impl(out, **params)` を順次適用。
- 重要: engine.core 内で registries を一切 import しない。

2) レジストリの純化（ドメイン純度の向上）
- shapes.registry / effects.registry:
  - 登録対象は「元の純関数（orig_impl）」に限定。
  - 取得 `get_*` は orig_impl を返す（Lazy ラッパを返さない）。
  - `__param_meta__` 等のメタはそのまま関数にぶら下げる。
  - 可能なら engine 依存（Geometry/LazyGeometry import）を排除（型は `Any`/Protocol/文字列注釈）。

3) API 層での解決と注入（唯一の registries 参照）
- api.shapes（G）:
  - `fn = shapes.registry.get_shape(name)` で orig_impl を取得。
  - ランタイム/量子化解決後、`LazyGeometry(base_payload=(fn, params))` を生成。
- api.effects（E.pipeline）:
  - `fn = effects.registry.get_effect(name)` で orig_impl を取得。
  - `Pipeline.__call__`（Lazy への plan 連結）では `plan += [(fn, params)]` として関数参照を積む。
  - Geometry 渡し時は `fn(geom, **params)` を即時適用。
- api.lazy_signature:
  - 署名生成は「関数 ID + 量子化 params」のみで構成（名前には依存しない）。
  - 関数 ID は `(impl.__module__, impl.__qualname__)` を基本とし、必要に応じて `id(impl)` を補助キーに使用。

4) キャッシュキー（core 側）
- shape 結果 LRU: `key = (impl_id, params_tuple)` に変更。
- prefix LRU: `key = (shape_key, [(impl_id, params_tuple), ...])`。
- 量子化は `common.param_utils.params_signature` を利用（engine.core → common は可）。

5) 互換性/移行
- 外部 API（`from api import G, E, shape, effect`）は維持。
- registries の「関数呼び出しで Lazy を返す」挙動は廃止（ドメイン純化）。
  - テストは `.realize()` 前提か、`api` 経由に統一。

---

## ディレクトリ/ファイル構成（変更案）

- 維持: `src/engine/core/lazy_geometry.py`（実体は上記仕様に更新）
- 新設（任意）:
  - `src/engine/core/cache.py`（LRU 群を分離し見通しを向上）
  - `src/common/func_id.py`（impl ID 抽出ヘルパを共通化）
- registries は現位置のまま（shapes/registry.py, effects/registry.py）。

---

## 実施手順（段階的）

Phase 0: 設計合意
- [x] 本計画の承認（依存方向/データ設計/互換ポリシー）

Phase 1: 最小分離（core → registries の遮断）
- [x] `engine.core.lazy_geometry` を関数参照ベースに改修（名前解決コードを削除）
- [x] 署名/キャッシュを関数参照ベースに移行
- [x] 既存ユニットテストで core の回帰がないか限定実行

Phase 2: API 注入化
- [x] `api.shapes` を orig_impl 注入に変更（`get_shape`→impl, Lazy 構築）
- [x] `api.effects` を orig_impl 注入に変更（plan に impl を積む）
- [x] `api.lazy_signature` を impl ベースに変更
- [x] smoke を実行し、動作確認（アーキテクチャテストも緑）

Phase 3: registry 純化
- [x] `shapes.registry`/`effects.registry` から Lazy/Geometry 依存を除去（必要なら後続）
- [x] デコレータは orig_impl を登録し、そのまま返す実装に簡素化（互換性確認後）
- [x] 影響するテストの更新（registry 直叩きは `api` へ寄せる or `.realize()` を介す）

Phase 4: アーキテクチャテスト緑化
- [x] `tests/test_architecture.py` を実行し、禁止エッジ/循環ゼロを確認
- [x] `architecture.md` を更新（依存方向・責務境界・署名/キャッシュ設計）

Phase 5: 仕上げ
- [ ] 変更ファイルに対する ruff/black/isort/mypy
- [ ] `pytest -q -m "not optional"`（合意あれば）
- [ ] リリースノート/移行ガイド（registry の Lazy 返却廃止についてテスト影響を記載）

---

## 受け入れ基準（DoD）

- engine.core が `effects.registry`/`shapes.registry` を一切参照しない。
- `tests/test_architecture.py::test_architecture_import_rules` が緑。
- smoke 一式が緑。公開 API は現行通り（G/E/shape/effect）。
- `architecture.md` が実装と同期。

---

## リスクと緩和

- 互換性: registries の挙動簡素化により、一部テストの呼び出し方変更が必要。
  - 緩和: `api` 経由の利用に統一。`.realize()` の明記。
- キャッシュキー: 関数参照の ID が環境依存になる可能性。
  - 緩和: `(__module__, __qualname__)` を基本 ID に、`id(impl)` を補助に用いて衝突回避。署名は量子化済み params を使用。
- 段階移行の複雑化: Phase を分けて影響範囲を限定しつつ進める。

---

## 確認事項（要回答）

1) registries を「純関数レジストリ」に簡素化（Lazy 返却の廃止）してよいか。
2) `engine.core.cache` などの小分割（任意）を実施してよいか。
3) キャッシュキーの impl ID 仕様（`module/qualname` + 量子化 params）で問題ないか。

承認後、詳細設計（関数シグネチャ、型注釈、キー構造）と具体的変更差分案を提示します。
