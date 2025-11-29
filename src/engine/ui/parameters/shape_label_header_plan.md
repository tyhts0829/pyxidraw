# Parameter GUI shape ヘッダラベル不正バグ調査 & 改善計画（ドラフト）

対象:

- バグ再現スケッチ: `sketch/251129.py`（`v_text` / `h_text`）
- 関連モジュール:
  - `src/api/shapes.py`（`G.label("...").text()` 経由の shape ラベル指定）
  - `src/engine/core/lazy_geometry.py`（`LazyGeometry.label()` と `ShapeLabelHook`）
  - `src/engine/ui/parameters/runtime.py`（`ParameterRuntime.relabel_shape()` / `_assign_shape_label()`）
  - `src/engine/ui/parameters/state.py`（`ParameterContext` / `ParameterDescriptor`）
  - `src/engine/ui/parameters/dpg_window_content.py`（カテゴリ別ヘッダ構築）

## 現状の挙動とバグ内容

- 期待仕様（AGENTS.md / architecture.md より）
  - Parameter GUI の shape ヘッダは `G.<shape>()` の呼び出しごとに分割される。
  - 同一 shape 名の場合、既定ヘッダは `text`, `text_1`, `text_2`, ... のように連番サフィックスが付く。
  - `G.label("title").text(...)` のようにラベル指定があれば、`title`, `title_1`, ... のように「ラベル名ベース」でヘッダ名が付与される。
- 現状のバグ:
  - `sketch/251129.py` で `v_text` / `h_text` をそれぞれ `G.text().label(uid="...")` でラベル付けしても、Parameter GUI 上では両方のパラメータが `v_text` ヘッダにまとまって表示される。
  - 期待する挙動は、`v_text` と `h_text` がそれぞれ独立したヘッダとして表示されること。

## 技術的な原因分析

### 1. shape 呼び出しと既定カテゴリの決定

- `ParameterRuntime.before_shape_call()` が shape 呼び出しをフックし、`ParameterContext` を生成している。
- `ParameterContext.category` により、shape のカテゴリ（ヘッダ名）が決まる:
  - scope=`"shape"` の場合、`name` と `index` から `text`, `text_1`, `text_2`, ... を生成。
  - `descriptor.id` には `"shape.<name>#<index>.<param>"` 形式が使われる。
- この段階では、ラベル指定が無い限り、同一 shape 名でも呼び出しごとに別カテゴリになる設計は概ね仕様通り。

### 2. shape ラベル指定のフロー

- `src/api/shapes.py`:
  - `G.label("title").text(...)` の場合:
    - `_ShapeCallContext` に `label="title"` を保持。
    - `G.text()` 呼び出し時に `runtime.before_shape_call("text", ...)` が走り、`index` に応じたカテゴリ（`text`, `text_1`, ...）で Descriptor を登録。
    - その後、LazyGeometry インスタンスに対して `lazy.label(ctx_label)` が呼ばれる。
  - `G.text().label("title")` の場合:
    - `G.text()` 側で上記と同様に `before_shape_call()` が走り、Descriptor が登録される。
    - その後、`LazyGeometry.label("title")` が直接呼ばれる。
- `src/engine/core/lazy_geometry.py`:
  - `LazyGeometry.label()` は shape ベース情報から `shape_name` を取得し、登録済みの `ShapeLabelHook` を通じて UI 層へ通知する。
  - フックのシグネチャは `ShapeLabelHook = Callable[[str, str], None]`（`shape_name`, `base_label`）。
- `src/engine/ui/parameters/runtime.py`:
  - モジュールロード時に `set_shape_label_hook(_shape_label_hook)` でフックを登録。
  - `_shape_label_hook(shape_name, base_label)` 内で `get_active_runtime()` を通じて現在の `ParameterRuntime` を取得し、`runtime.relabel_shape(shape_name, base_label)` を呼び出す。

### 3. `relabel_shape()` 実装とカテゴリ上書き

- `ParameterRuntime.__init__` では、shape 用のラベル管理として以下のフィールドを持つ:
  - `_shape_label_by_name: dict[str, str]`
  - `_shape_label_counter: dict[str, int]`
- `_assign_shape_label(shape_name, base_label)` の現在の挙動:
  - `key = shape_name`（例: `"text"`）をラベル辞書のキーとして採用。
  - すでに `key` が登録されている場合、`label_map[key]` をそのまま返す（ラベル再計算しない）。
  - 未登録の場合のみ、`_shape_label_counter[base_label]` をインクリメントして `base_label` or `base_label_2` のような表示ラベルを生成し、`label_map[shape_name]` に保存。
- `relabel_shape(shape_name, base_label)`:
  - 上記 `_assign_shape_label` で算出された `display_label` を用い、`ParameterStore.update_descriptors()` によって Descriptor 群の `category` を一括上書きする。
  - 上書き対象は:
    - `desc.source == "shape"`
    - `desc.id` を `"."` で 2 分割した第 2 要素（例: `"text#0"`）が `f"{shape_name}#"` で始まるもの（= 指定 shape 名に属するすべての呼び出し）。

### 4. v_text / h_text が同一ヘッダにまとまる理由

- `sketch/251129.py` のフロー（shape 名はどちらも `"text"`）:
  1. `G.text().label("v_text")` 呼び出し:
     - `before_shape_call("text", ...)` で index=0 → Descriptor.category="text" で登録。
     - 直後に `LazyGeometry.label("v_text")` → `_shape_label_hook("text", "v_text")` → `relabel_shape("text", "v_text")`。
     - `_assign_shape_label` で `key="text"` かつ未登録のため、`display_label="v_text"` を生成し、`_shape_label_by_name["text"]="v_text"` として保存。
     - `relabel_shape` が `shape.text#0.*` に属する Descriptor の `category` をすべて `"v_text"` に書き換え。
  2. `G.text().label("h_text")` 呼び出し:
     - `before_shape_call("text", ...)` で index=1 → Descriptor.category="text_1" で登録。
     - `LazyGeometry.label("h_text")` → `_shape_label_hook("text", "h_text")` → `relabel_shape("text", "h_text")`。
     - `_assign_shape_label` は既に `key="text"` が登録済みのため、`display_label` として `"v_text"` を返す（`base_label` は無視される）。
     - `relabel_shape` が `shape.text#0.*` と `shape.text#1.*` の両方を `category="v_text"` へ上書き。
- 結果として、`v_text` と `h_text` のパラメータがどちらも `v_text` ヘッダに集約されてしまう。
- 根本原因:
  - shape ラベル管理が `shape_name` 単位（例: `"text"`）でしか行われず、「同一 shape 名の複数呼び出しに対して別々のラベルを割り当てる」設計になっていないこと。
  - `relabel_shape` が、指定された shape 名に属するすべての呼び出し（index 0,1,2, ...）を一括で同一カテゴリに書き換えていること。

## 仕様整理と改善方針（高レベル）

### A. 仕様の整理（どうあるべきか）

- 既定挙動:
  - 同一 shape 名でラベル指定が無い場合、`G.text()` の呼び出しごとに `text`, `text_1`, `text_2`, ... のようにヘッダを分割する（現在の `ParameterContext.category` の仕様通り）。
- shape ラベル指定の挙動:
  - `G.label("title").text(...)` または `G.text().label("title")` のいずれでも、「そのラベル指定が紐づく 1 回の shape 呼び出し」に対して、カテゴリ名として `title`, `title_1`, ... を割り当てる。
  - 同一ラベル文字列が複数回使われた場合は、`title`, `title_1`, `title_2` のようにラベル単位で連番が付く。
  - 異なるラベル（例: `v_text` / `h_text`）は独立したカテゴリとして扱い、それぞれ別のヘッダに表示される。
- 互換性:
  - 現状のテスト `tests/ui/parameters/test_shape_label_chain.py` が保証するのは「単一ラベル `title` を 1 回適用したとき、`shape.text#0.*` のカテゴリが `title` になること」のみ。
  - 既存テストが想定している挙動（単一ラベルケース）は維持したまま、同一 shape 名に対する複数ラベルのケースを拡張する。

### B. 実装レベルの改善方針

- ラベル管理の粒度を「shape 名」単位から「shape 呼び出し（index）＋ラベル」単位へ細分化する。
- `relabel_shape()` が「指定 shape 名に属するすべての呼び出し」を一括で書き換えるのではなく、「ラベルを適用すべき 1 件（またはラベル単位の 1 グループ）」に限定してカテゴリを書き換えるようにする。
- ラベル適用の順序は、以下のように単純なルールで決める:
  - `LazyGeometry.label()` が呼ばれた時点で、そのフレーム内に存在する「同一 shape 名の呼び出し」のうち、「まだラベル適用されていない（カテゴリが既定値のまま）のインスタンス」に対して順番に割り当てる。
  - ラベル適用済みかどうかは、Descriptor.category が既定パターン（`text`, `text_1`, ...）のままかどうか、あるいは内部状態（ラベル適用済みフラグ）で判定する。

## 改善アクションチェックリスト（提案）

### 1. 仕様と境界条件の明確化

- [ ] AGENTS.md / architecture.md の Parameter GUI セクションを読み直し、「shape ヘッダとラベルの関係」について現在のコードと乖離している点を洗い出す。
- [ ] `G.label("title").text(...)` と `G.text().label("title")` の 2 パターンを正式にサポート対象とするか確認する（どちらも現状動作しているため、互換性維持が望ましいと考えています）。
- [ ] ラベル名が shape 名と同じ場合（例: `G.label("text").text()`）の扱いを確認する（既定カテゴリとの衝突時の優先度ルールを決める）。

### 2. 実装変更案（概要設計）

- [x] `ParameterRuntime` の shape ラベル管理を見直し、`_shape_label_counter` を「ラベル文字列に対する連番管理」のみに簡素化しつつ、`_shape_label_assigned: set[tuple[str, int]]` で「shape 名＋ index」単位の割り当て済み状態を管理するようにした。
- [x] `relabel_shape(shape_name, base_label)` 内で、「どの index の呼び出しにラベルを適用するか」を決定するロジックを実装した。`ParameterStore.descriptors()` から `shape_name` に一致する Descriptor を一覧取得し、`index` ごとにグルーピングして「まだラベル適用されていない index」を昇順に選ぶ。すべて割り当て済みの場合は最大 index に対して上書きするフォールバックとした。
- [x] `_assign_shape_label(shape_name, base_label)` の責務を「ラベル文字列に対する連番管理（`title`, `title_1`, ...）」に限定し、どの index に適用するかの判断は `relabel_shape` 側で行うように変更した。
- [x] `relabel_shape()` 内の Descriptor 更新処理を、「対象 index のみを filter する」ように変更した。既存の `shape.text#0.*` / `shape.text#1.*` などの id パターンに基づき、`index` を抽出して比較している。

### 3. テストの追加と既存テストの維持

- [x] `tests/ui/parameters/test_shape_label_chain.py` に対して、以下のケースを追加した: `G.label("v_text").text()` と `G.label("h_text").text()` を同一フレーム内で呼び出し、それぞれが別カテゴリ（`"v_text"` / `"h_text"`）になることを検証するテスト、および `G.text().label("v_text")` と `G.text().label("h_text")` のパターンでも同様の挙動になることを検証するテスト。
- [x] 追加テストケースで、index ごとのカテゴリ名が期待通りに変化していることを確認した（`shape.text#0.*` → `v_text`, `shape.text#1.*` → `h_text`）。
- [ ] バグ報告スケッチ `sketch/251129.py` のケースを簡略化したユニットテスト（shape/text のみ）を追加するか検討する（現時点では `test_shape_label_chain.py` の拡張で再現できているため、追加は保留中）。

### 4. パフォーマンスと安全性の確認

- [ ] `relabel_shape()` 内で `ParameterStore.descriptors()` をフルスキャンする場合のオーバーヘッドを評価し、descriptor 数が多い場合でも実用的な範囲に収まるか確認する。
- [ ] 例外処理方針を整理し、ラベル適用で予期せぬ例外が発生した場合でも Parameter GUI 全体が崩壊しないよう、現状のフェイルソフト設計を維持する。
- [ ] ラベル適用ロジックが hud/style/palette など他カテゴリ（source != "shape"）に影響を与えないことを再確認する。

### 5. ドキュメントと仕様同期

- [ ] `architecture.md` の Parameter GUI セクションに、「同一 shape 名に異なるラベルを付与した場合のヘッダ挙動」を明記する（`v_text` / `h_text` のような例を簡単に追加）。
- [ ] 必要に応じて `AGENTS.md` の Parameter GUI ルールも更新し、「ラベル名ベースのヘッダ分割」と「ラベル未指定時の既定名（shape 名ベース）」の優先順位を明文化する。
- [ ] 今回の改善で追加/変更されるテストケースを簡潔にコメントしておき、将来のリファクタ時に仕様が壊れた場合の検知ポイントとする。

### 6. 実装・検証手順（作業フロー）

- [x] 上記仕様・変更案について、影響範囲（`ParameterRuntime` / `LazyGeometry.label` / Parameter GUI テスト）を踏まえた上で実装方針を確定した。
- [x] `ParameterRuntime` の shape ラベル管理ロジックをリファクタリングし、`relabel_shape()` の挙動を更新した。
- [x] 新規/既存テスト（主に `tests/ui/parameters/test_shape_label_chain.py`）を実行し、`pytest -q tests/ui/parameters/test_shape_label_chain.py` でグリーンになることを確認した。
- [ ] 必要に応じて `sketch/251129.py` を実行し、Parameter GUI 上で `v_text` / `h_text` が期待通り別ヘッダに表示されることを目視で確認する。
- [ ] 変更内容とテスト結果を簡潔にまとめ、architecture.md / AGENTS.md の更新とあわせて差分を整理する。

## 確認してほしい点

- 上記の方針で「同一 shape 名に対して複数ラベルを正しく分離する」実装変更を進めてもよいか。
- ラベルと index の対応付けロジックについては、a)「LazyGeometry.label が呼ばれた順番」ベースで、まだラベルが適用されていない index に順番に割り当てる方針で進めてよいか。
- テスト追加の粒度（特に `sketch/251129.py` 相当の挙動をどこまでユニットテストで再現するか）について、希望があれば教えてください。

※ このファイルは「バグ調査結果と改善計画（ドラフト）」です。実際のコード変更は、上記チェックリスト内容についてご確認いただいた後の合意に基づいて行います。
