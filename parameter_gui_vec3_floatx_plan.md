# Parameter GUI: Vec3 を float スライダ3本で扱う — クリーン実装計画（破壊的変更前提）

目的: 既存の「成分ごとに別 Descriptor を発行する」設計を廃止し、Vec3 を「単一の vector パラメータ」として一次情報化。GUI は1行に x/y/z の3スライダ（float×3）を水平配置して編集する。内部状態・同期・API すべてをシンプルにする。

## 採用方針（破壊的変更点）
- Vector は「親 Descriptor 1 つ」に統一（成分ごとの Descriptor を撤廃）。
- Range ヒントは Vector 専用の `VectorRangeHint` を導入（min/max/step を各成分で保持）。
- ParameterStore は親 ID（例: `effect.rotate#1.angles_rad`）で tuple 値を保持・上書きする（成分 ID は廃止）。
- ParameterWindow は vector1行で 3 スライダ（x/y/z）を生成し、親 ID の値を合成して Store を更新。
- ValueResolver は vector 既定値採用時に「親 Descriptor」を1件のみ登録。提供値ありのときは GUI 登録しない（従来ルール維持）。
- 既存テスト（成分 ID を期待）は更新する。

## 設計概要
- 型/メタ
  - ParameterDescriptor
    - `value_type`: `"float" | "int" | "bool" | "enum" | "vector"`
    - `default_value`: vector の場合は `tuple[float, float, float]`
    - `range_hint`: scalar 用（従来）。vector の場合は `None`。
    - `vector_hint`: `VectorRangeHint | None`（新設）
  - VectorRangeHint（新設）
    - `min_values: tuple[float, float, float]`
    - `max_values: tuple[float, float, float]`
    - `steps: tuple[float | None, float | None, float | None]`
- 値解決（src/engine/ui/parameters/value_resolver.py）
  - `_resolve_vector()` を簡素化: 親 Descriptor を1件だけ発行し、`vector_hint` を設定。Store.register → Store.resolve も親 ID で実施。
  - 既存の成分分解ロジック・`_register_vector()` は削除。
- ストア（src/engine/ui/parameters/state.py）
  - 既存 API のまま（original/override は Any）。vector は tuple をそのまま保持。クランプ/量子化はしない。
  - `RangeHint` はそのまま存置（scalar 用）。`VectorRangeHint` を追加（新規型）。
- GUI（src/engine/ui/parameters/dpg_window.py）
  - `value_type=="vector"` の分岐で 1 行に 3 スライダを生成（x,y,z）。
  - スライダ ID は内部タグ（例: `f"{parent_id}::x"`）を用い、`user_data=(parent_id, axis_index)` で親に紐付け。
  - `_on_widget_change()` は `(parent_id, idx)` を受け取り、Store の親値 tuple を取り出して成分差し替え→ `set_override(parent_id, new_tuple)`。
  - `_on_store_change()` は親 ID を受けたら、内部タグ群へ各成分値を `dpg.set_value()` で反映。
  - Range は `vector_hint` の各成分を使用（無い場合は 0..1 を既定）。

## 影響（Breaking changes）
- Store/Descriptor から成分 ID（`.x/.y/.z`）が消え、親 ID のみに統一。
- `tests/ui/parameters/test_value_resolver.py::test_parameter_value_resolver_handles_vector_params_defaults_register_gui` の期待を変更（親 ID 登録の確認に更新）。
- UI 出力構造（タグ構成）が変わるが、外部 API（公開 API 層）には影響しない。

## 実装詳細（手順）
1) 型追加/変更
   - state.py
     - `VectorRangeHint`（新規 dataclass）を追加。
     - `ParameterDescriptor` に `vector_hint: VectorRangeHint | None` を追加し、`value_type=="vector"` のときだけ使用。
     - `RangeHint`（既存）は scalar 用として据え置き。
     - `ParameterLayoutConfig` に `derive_vector_range(dim=3)` を追加（既定は各成分 0..1）。
2) 値解決の刷新
   - value_resolver.py
     - `_resolve_vector()` を親 Descriptor 登録に変更。`_register_vector()` を削除。
     - `param_meta` の `min/max/step` がベクトルのとき `VectorRangeHint` を組み立て。
3) GUI 刷新
   - dpg_window.py
     - `_create_widget()` の `vt=="vector"` 分岐を、`dpg.add_slider_float` を3本並列に生成する実装へ差し替え。
     - 内部タグ（`parent_id::x|y|z`）を用いた更新・同期ハンドリングを実装。
     - `mount()` は従来通り Descriptor を 1 件ずつ行化（グルーピングは不要になる）。
4) 既存ロジックの清掃
   - `vector_group` フィールドは廃止予定（将来の親子表現に不要）。まずは未使用化→後段で削除。

## 追加・変更ファイル（想定）
- 変更: `src/engine/ui/parameters/state.py`（`VectorRangeHint` 追加、`ParameterDescriptor` 拡張、`ParameterLayoutConfig` に vector 既定追加）
- 変更: `src/engine/ui/parameters/value_resolver.py`（vector の親 Descriptor 登録・簡素化）
- 変更: `src/engine/ui/parameters/dpg_window.py`（vector UI を float×3 スライダで表示・同期）
- 更新: `architecture.md`（Vec3 の扱い方と根拠を追記。参照: `state.py`, `value_resolver.py`, `dpg_window.py`）
- 更新: 近接 AGENTS.md（parameters 配下）の “vector は x/y/z/w に分割” 記述をアップデート（親 Descriptor 一体化へ）

## テスト計画（編集ファイル優先の高速ループ）
- 更新: `tests/ui/parameters/test_value_resolver.py::test_parameter_value_resolver_handles_vector_params_defaults_register_gui`
  - 期待: `effect.rotate#1.angles_rad`（親 ID）が Descriptor 登録されること。
  - 値: 既定 tuple が返ること。
- 追加: `tests/ui/parameters/test_vector_widget.py`
  - DPG 未導入環境でのスタブ下でも API 例外が出ないこと（smoke）。
  - Store の親 ID override により内部タグ（`::x|y|z`）スライダ値が更新されること（`_on_store_change` 経由の間接検証）。

実行コマンド（変更ファイル限定）:
- Lint: `ruff check --fix src/engine/ui/parameters`
- Format: `black src/engine/ui/parameters && isort src/engine/ui/parameters`
- TypeCheck: `mypy src/engine/ui/parameters`
- Test: `pytest -q tests/ui/parameters -k vector`

## 具体的作業チェックリスト
- [ ] state: `VectorRangeHint` を追加、`ParameterDescriptor` に `vector_hint` を追加
- [ ] state: `ParameterLayoutConfig.derive_vector_range()` を追加
- [ ] resolver: `_resolve_vector()` を親 Descriptor 登録に刷新（成分登録の撤廃）
- [ ] resolver: `param_meta(min/max/step)` から `VectorRangeHint` を構築
- [ ] dpg: vector 行にスライダ3本を水平配置し、親 ID の値を合成更新
- [ ] dpg: `_on_store_change()` で親 ID 通知から内部タグへ反映
- [ ] 清掃: `vector_group` の未使用化（後続で除去）
- [ ] テスト更新/追加（上記）
- [ ] ruff/black/isort/mypy（変更ファイルのみ）
- [ ] `pytest -q tests/ui/parameters` 緑
- [ ] `architecture.md`/AGENTS を実装と同期

## 既定判断（確認不要の動作）
- スライダのラベルは右列の各軸に小さく `x/y/z` を表示。
- ステップは UI スライダの仕様に従い、`VectorRangeHint.steps[i]` が指定されていればその粒度で操作（丸めは UI 任せ）。Store は実値をそのまま保持し、追加の量子化は行わない。
- Vec4 は同実装で水平4本（`x,y,z,w`）。`VectorRangeHint` は 3/4 成分に対応。
- 既定レンジは各軸 0..1。`__param_meta__` があれば実レンジを尊重。

## 完了条件
- 変更ファイルに対する `ruff/black/isort/mypy` が成功。
- `pytest -q -m smoke` および `tests/ui/parameters` が緑。
- `architecture.md` と実装の差分ゼロ（参照箇所明記）。

---

この方針で実装に移行してよければ、そのまま着手します。必要に応じて段階的に PR を分割します（型/Resolver → GUI → 清掃/Docs）。
