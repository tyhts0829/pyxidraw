Lazy + Lazy 実装のアーキテクチャ違反修正計画
==========================================

背景
----
- smoke 実行で `tests/test_architecture.py::test_architecture_import_rules` が失敗。
- 失敗内容:
  - 禁止層の依存: `engine.core.lazy_geometry` → `api.lazy_signature`
  - 循環参照（モジュールレベル）: `api.lazy_signature` ↔ `engine.core.lazy_geometry`
  - 自己 import 検出: `engine.core.lazy_geometry` → `engine.core.lazy_geometry`

目的
----
- Core 層から API 層への依存を排除し、循環参照を解消。
- 既存の Lazy+Lazy の `+` 挙動（遅延・集約・O(totalN)）を維持。
- 署名（キャッシュキー）の安定性を保ち、Prefix LRU のヒット率を維持。

設計方針
--------
1) 署名生成の自己完結化（Core 内）
- `api.lazy_signature` に依存せず、`engine.core.lazy_geometry` 内で LazyGeometry の署名（ハッシュ可能タプル）を構築。
- 既存ユーティリティを活用:
  - `common.func_id.impl_id`（関数ID）
  - `common.param_utils.params_signature`（量子化＋ハッシュ化可能化）
- 署名構成案（タプル）:
  - base: `("geom-id", id(g))` または `("shape", impl_id(shape_impl), params_signature(shape_impl, params))`
  - plan: `[(impl_id, params_signature(impl, filtered_params)), ...]`
  - 最終: `(base, tuple(plan))`
- フィルタ: `_ref` を含むキーと `ObjectRef` 値は署名から除外（実体参照をキャッシュキーに含めない）。

2) 自己 import の排除（duck typing）
- `_fx_concat_many` 内の `from engine.core.lazy_geometry import LazyGeometry as _LG` を廃止。
- 判定は duck typing に変更:
  - `obj` が `Geometry` ならそのまま
  - それ以外で `hasattr(obj, "realize")`/`hasattr(obj, "plan")` を満たすなら `obj.realize()` を呼ぶ

3) 外部 API 非変更・性能維持
- `LazyGeometry.__add__` の公開挙動（Lazy 同士のみ許容、遅延・集約）は据え置き。
- `_fx_concat_many` は 1 パス結合（事前見積→一括確保→ブロックコピー）を維持。

実施手順（チェックリスト）
--------------------------
- [x] `engine.core.lazy_geometry`: `__add__` から `api.lazy_signature` 依存を排除（import 削除）。
- [x] `engine.core.lazy_geometry`: `LazyGeometry` 用の内部署名ビルダ `_lazy_sig_for(lg)` を追加。
- [x] `_fx_concat_many`: 自己 import を削除し、duck typing で Lazy/Geometry を正規化。
- [x] 署名生成時の実体参照除外フィルタ（`*_ref` キー・`ObjectRef` 値）を導入済み（再確認のみ）。
- [ ] ruff/mypy を変更ファイルに対して実行。
- [ ] `pytest -q -m smoke` を実行し、`tests/test_architecture.py` の緑化を確認。
- [ ] `architecture.md` に「Core→API 依存禁止」「Lazy 署名のCore内完結化」を追記。

検証
----
- 機能: `sketch/251111.py`（Lazy のみの和）で描画されること。
- 性能: `a + b + c + ...` でフレーム落ち・GC スパイクがないこと（目視／簡易計測）。
- キャッシュ: HUD の Prefix LRU が MISS→HIT に遷移すること（有効時）。

リスクと対策
------------
- 署名定義の差異でキャッシュヒット率が変動する
  - 対策: 既存 `params_signature` を一貫利用し、impl_id と組み合わせた決定的タプルに限定。
- duck typing 判定の誤検出
  - 対策: `Geometry` 優先の `isinstance` 判定→`hasattr(realize, plan)` の順に絞り込む。

補足（非目標）
--------------
- `Lazy + Geometry`/`Geometry + Lazy` は今回非対応のまま。
- `__iadd__`（`+=`）は見送り。
