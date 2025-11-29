# parameter_gui shape ヘッダ分割計画（ドラフト）

目的: Parameter GUI 上で、同じ shape（例: `G.text`）が複数回呼ばれている場合でも、呼び出しごとに別のヘッダ（カテゴリ）として表示し、個々のインスタンス単位でパラメータを把握・調整しやすくする。

## 現状と課題

- 現状:
  - `main.py` では `G.text(...)` が 2 回登場するが、Parameter GUI の shape セクションではヘッダが `text` と `polygon` の 2 つのみになり、2 つの text のパラメータが 1 つの `text` ヘッダにまとまって表示される。
  - shape パラメータの Descriptor ID 自体は `shape.text#0.*`, `shape.text#1.*` のようにインデックス付きで分かれているが、`category` がいずれも `"text"` のため、UI 上のヘッダ単位では区別されていない。
- 課題:
  - `G` 呼び出しごとに異なる役割の shape を使っている場合でも、ヘッダが 1 つにまとまるため、どのヘッダがどの呼び出しに対応しているか直感的に分かりづらい。
  - 特に `G.text` のように同一 shape を複数回使う場合、ヘッダ単位で「この text はどのジオメトリか」を切り分けたい。

## 想定仕様（ドラフト）

- shape ヘッダの分割:
  - `G.<shape>()` が呼ばれるたびに、Parameter GUI 上では「shape カテゴリのヘッダ」を別インスタンスとして表示する。
  - 同じ shape 名でも呼び出し回数ごとに別ヘッダになる（例: `G.text` を 2 回呼ぶと `text` ヘッダと `text_1` ヘッダが現れる）。
- ヘッダ名の命名規則:
  - shape 名を `name` としたとき、1 回目の呼び出しに対応するヘッダ名は `name`（例: `text`）。
  - 同じ `name` の 2 回目以降の呼び出しでは、`name_1`, `name_2`, ... のように末尾に `_1`, `_2`, ... を付与する。
  - 付与する番号は「フレーム内での同一 shape 名の呼び出し順」に基づく。
- カウントとリセット:
  - shape ごとの呼び出し回数は `ParameterRuntime.begin_frame()` 呼び出し時にリセットされ、各フレームで同じ順序で呼び出される限り、ヘッダ名も安定する。
- 互換性:
  - Descriptor ID（`shape.<name>#<index>.<param>`）と `category_kind`（shape/pipeline 等）は従来通りとし、`category` の値（= ヘッダ表示名）のみを変更する。
  - effect/pipeline 側のカテゴリ決定（`.label()` からの `poly_effect_1`, `poly_effect_2` 等）は現行仕様を維持する。

## 実装方針（案）

- shape 呼び出しインデックスの利用:
  - すでに `ParameterRuntime.before_shape_call()` 内で `ParameterRegistry` を用いて shape 名ごとの出現回数（0 ベース）を計測し、その値を `ParameterContext.index` に渡している。
  - 新規の状態やレジストリは増やさず、この `index` をそのまま shape カテゴリ名の決定に利用する。
- カテゴリ名の決定ロジック:
  - `engine.ui.parameters.value_resolver.ParameterContext.category` の shape 分岐を変更し、`index == 0` のときは `name`、`index >= 1` のときは `f"{name}_{index}"` を返す。
  - これにより、1 回目 `text`、2 回目 `text_1`、3 回目 `text_2` ... というヘッダ名になる（`ParameterRegistry` が 0,1,2,... を返す前提）。
- GUI 側のグルーピング:
  - `ParameterWindowContentBuilder._build_grouped_table()` は `(category_kind, category)` 単位で Descriptor をグルーピングしているため、`category` が呼び出しごとに異なれば自動的にヘッダも分かれる。
  - GUI 実装側には新たな状態を追加せず、`ParameterDescriptor.category` の値のみで制御する。
- 他機能への影響:
  - 上記変更は shape の `category` のみを変更し、`category_kind` や Descriptor ID、永続化キー（JSON 内の `overrides` キー）には影響しないため、既存の override/persistence ロジックはそのまま動作する。
  - effect/pipeline のカテゴリ名決定（`ParameterRuntime._assign_pipeline_label` + `pipeline_label`）には手を入れない。

## やること（チェックリスト）

### 1. 仕様・挙動の確定

- [ ] shape ヘッダ名の命名規則を文書として確定する（1 回目 `name`, 2 回目以降 `name_1`, `name_2`, ... でよいかを合意する）。
- [ ] フレームごとのカウントリセットを `ParameterRuntime.begin_frame()` で行う前提が妥当か確認する（既存のフレーム設計と矛盾しないか）。
- [ ] shape ごとに任意の表示名を付ける API（例: `G.text(label="title")` のようなもの）を現時点では導入しない、という方針で問題ないか確認する。

### 2. コア実装の変更

- [ ] `engine.ui.parameters.value_resolver.ParameterContext.category` を修正し、`scope == "shape"` の場合に `index` に応じてカテゴリ名を切り替えるようにする（`index == 0` なら `name`、それ以外は `f"{name}_{index}"` のような実装）。
- [ ] `engine.ui.parameters.runtime.ParameterRuntime.before_shape_call` で供給している `index` が `ParameterRegistry.next_index(shape_name)` に基づく 0 ベース連番であることを確認し、想定どおり `text`, `text_1`, `text_2` … となることをコードレベルで検証する。
- [ ] shape 以外（effect/pipeline/hud/style/palette 等）のカテゴリ決定ロジックに変更が波及していないことを確認する（`ParameterContext.category` の effect 分岐と `ParameterRuntime._assign_pipeline_label` 周りを再確認）。

### 3. テスト追加・更新

- [ ] `tests/ui/parameters/test_value_resolver.py` もしくは新規テストファイルを追加し、`scope="shape"` の `ParameterContext` を用いたユニットテストで `ParameterDescriptor.category` がインデックスに応じて `text`, `text_1`, `text_2` となることを検証する。
- [ ] effect 側のカテゴリ名（`poly_effect_1`, `poly_effect_2` 等）が変更されていないことを確認するテストを用意するか、既存テストで十分かどうかレビューする。
- [ ] shape ヘッダ名の変更が `persistence.save_overrides` / `load_overrides` の挙動に影響しないことを確認する（Descriptor ID ベースで動作していることをテストまたはコードレビューで明示する）。

### 4. Parameter GUI 実装との整合確認

- [ ] `src/engine/ui/parameters/dpg_window_content.py` の `_build_grouped_table` / `_flush_group` が、新しい shape カテゴリ名でも期待どおり「ヘッダごとにテーブルを分ける」だけの挙動になっていることを確認する（追加の状態や副作用がないことを確認）。
- [ ] `_style_owner_key` / `_style_label` など style 系ヘルパが shape カテゴリ名の変更で予期せず挙動を変えないことを確認する（shape 系 Descriptor が style グループに混ざらないことも含めて軽くチェック）。

### 5. ドキュメントと手動動作確認

- [ ] `architecture.md` の「パラメータ GUI」セクションに、shape カテゴリ名の決定規則（`G.<shape>()` の呼び出しごとにヘッダを分割し、同一 shape 名には `_1`, `_2` などのサフィックスを付ける）を追記する。
- [ ] 必要であれば、ルート `AGENTS.md` の Parameter GUI 関連の箇所に「shape ヘッダは呼び出しごとに分割される」旨を 1 行程度で追記するか検討する。
- [ ] `main.py` のように `G.text` を複数回呼び出すスケッチを `use_parameter_gui=True` で実行し、shape セクションのヘッダが `text`, `text_1`, `polygon` のように分割されていることを目視確認する。

## 確認してほしい点

- shape ヘッダ名のサフィックスを「2 回目以降 `_1`, `_2`, ...」とする案（1 回目は無印）で問題ないか、それとも 1 回目から `_0`, `_1`, ... のように数字を付けたいか。
- `G` 呼び出しごとにヘッダを完全に分割すると、同一 shape のパラメータが 1 つのヘッダにまとまらなくなりますが、現状のユースケースではこの方が望ましいと考えてよいか。
- 将来的に、shape ごとに任意の表示名（例: `G.text(label="title")`）を与えてヘッダ名に反映するような API 拡張を検討すべきか、それとも当面は「shape 名 + インデックス」のみで十分か。

※ このファイルは「shape ヘッダ分割の改善計画（ドラフト）」です。ここでの方針とチェックリストをご確認いただき、問題なければこの計画に沿って実装とテスト、ドキュメント更新を進めます。

