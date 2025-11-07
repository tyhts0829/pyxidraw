# テスト仕様更新計画（LazyGeometry 既定仕様へ整合）

目的: 形状 API（`G.<name>(**params)`）が既定で `LazyGeometry` を返す現行仕様に、テストコードを整合させる。機能仕様は変更しない（方針B）。

- 非対象: アーキテクチャ違反（`engine.core.lazy_geometry` → registries 依存）は本計画では変更しない。該当テスト（`tests/test_architecture.py`）は現状のまま維持し、後続タスクでコード側を修正する。

## 背景 / 現状

- 現行 API: `src/api/shapes.py` は `G.<name>(...) -> LazyGeometry` を返す（遅延が既定）。
- 一部テストは旧前提（即時 `Geometry` 返却）で書かれており、`isinstance(g, Geometry)` などで失敗。
- 失敗例（smoke）:
  - `tests/api/test_shape_import.py::test_api_shape_import_and_registration`
  - `tests/shapes/test_asemic_glyph_function.py::test_asemic_glyph_is_registered_and_generates_geometry`
- 将来的に full テストでも失敗が想定される箇所:
  - `tests/api/test_shapes_api.py`（複数）
  - `tests/shapes/test_registry_and_sphere.py`

## 変更方針（テスト側の整合パターン）

- 形状生成の戻り値に対する型前提の更新:
  - 旧: `g = G.xxx(...); assert isinstance(g, Geometry)`
  - 新: `g = G.xxx(...).realize(); assert isinstance(g, Geometry)`
- レジストリ API 経由（`shapes.registry.get_shape(name)`）の戻り値も遅延ラッパであるため、同様に `.realize()` を呼ぶ。
- インスタンス同一性テストの更新:
  - 旧: `a is b`（同一 `Geometry` インスタンスを前提）
  - 新: `a.realize() is b.realize()`（形状結果 LRU を通じて同一インスタンスであることを確認）

## 影響範囲（修正対象のテスト）

- tests/api/test_shape_import.py
- tests/shapes/test_asemic_glyph_function.py
- tests/api/test_shapes_api.py
- tests/shapes/test_registry_and_sphere.py

## タスク（チェックリスト）

- [x] tests/api/test_shape_import.py を Lazy 前提に更新
  - [x] `g = G.tmp_shape_smoke(...)` → `g = G.tmp_shape_smoke(...).realize()`
  - [x] `assert isinstance(g, Geometry)` は維持
  - [x] 登録解除処理（後片付け）は現状維持
- [x] tests/shapes/test_asemic_glyph_function.py を Lazy 前提に更新
  - [x] `g = G.asemic_glyph(...)` → `g = G.asemic_glyph(...).realize()`（両テスト）
  - [x] 生成物非空の検証は維持
- [x] tests/api/test_shapes_api.py を Lazy 前提に更新
  - [x] `test_dynamic_dispatch_and_geometry`: `g = G.sphere(...)` → `.realize()` 付与
  - [x] `test_shape_factory_lru_returns_same_instance`:
    - [x] `a is b` → `a.realize() is b.realize()` に変更
  - [x] `test_from_lines_and_empty` は `from_lines/empty` が `Geometry` を返すため変更不要（現状維持）
- [x] tests/shapes/test_registry_and_sphere.py を Lazy 前提に更新
  - [x] `g = fn(...)` → `g = fn(...).realize()`
  - [x] 型/オフセットの検証は維持

## 検証（編集ファイル優先の高速ループ）

- Lint/Format/Type（対象ファイルのみ）
  - `ruff check --fix tests/api/test_shape_import.py`
  - `ruff check --fix tests/shapes/test_asemic_glyph_function.py`
  - `ruff check --fix tests/api/test_shapes_api.py`
  - `ruff check --fix tests/shapes/test_registry_and_sphere.py`
  - `black tests/api/test_shape_import.py tests/shapes/test_asemic_glyph_function.py tests/api/test_shapes_api.py tests/shapes/test_registry_and_sphere.py && isort <同上>`
  - `mypy <対象ファイル>`（型注釈は変えないため実質スキップ可）
- テスト（対象限定）
  - `pytest -q tests/api/test_shape_import.py::test_api_shape_import_and_registration`
  - `pytest -q tests/shapes/test_asemic_glyph_function.py::test_asemic_glyph_is_registered_and_generates_geometry`
  - `pytest -q tests/api/test_shapes_api.py -q -k "dynamic_dispatch or lru or from_lines"`
  - `pytest -q tests/shapes/test_registry_and_sphere.py`
- スモーク再確認
  - `pytest -q -m smoke`

## 受け入れ条件（Done）

- 上記 4 ファイルのテストが緑。
- スモーク（`-q -m smoke`）が緑。
- スタブ同期系（G/pipeline）は変更なしのため引き続き緑。
- アーキテクチャテストの失敗は現時点では残存（後続タスク）。

## 後続タスク（別計画）

- アーキテクチャ違反の解消（コード側）
  - `engine.core.lazy_geometry` から `effects.registry` / `shapes.registry` の参照を排除。
  - 具体案（要設計レビュー）:
    - `LazyGeometry.base_payload` に「名前」ではなく「呼び出し可能（impl 参照）」を格納し、`api` レイヤで解決済みの関数を注入。
    - `plan` もエフェクト名ではなく impl 参照（＋署名用メタ）を持たせる。
    - これにより engine.core → registries の依存と import サイクルを解消。
  - 試験観点: `tests/test_architecture.py` のレイヤ/禁止エッジ/循環が全緑になること。

## リスク / 留意点

- realize() による実行コストがテストで増えるが、対象は限定的で影響軽微。
- `a.realize() is b.realize()` は形状 LRU キャッシュ前提のため、キャッシュ設定が無効化された環境変数設定時は恒等性が崩れる可能性あり（現状デフォルト有効）。

## 確認事項（要回答）

- このテスト更新方針（`.realize()` 呼び出し／恒等性の比較方法）で問題ありませんか？
- `test_shape_factory_lru_returns_same_instance` の恒等性要件を「同一実体（is）」で維持するか、それとも「幾何同値（座標一致）」へ緩和しますか？（現計画は前者）

---

実行準備が整い次第、この計画に従ってテスト修正を行い、進捗は本ファイルのチェックボックスを更新して共有します。
