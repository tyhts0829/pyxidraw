# shape label メソッドチェーン拡張計画（ドラフト）

目的: `G.label()` を shape チェーンの任意位置（`G.label("title").text(...)`, `G.text(...).label("title")` など）で呼び出せるようにし、Parameter GUI 上の shape ヘッダ名と整合した挙動にする。あわせて型スタブの警告（`Geometry` に `label` が無い）を解消する。

## 現状整理

- 実行時 API
  - `api.shapes.ShapesAPI` (`src/api/shapes.py`)
    - `G = ShapesAPI()` として公開。
    - `G.<shape>(...)` は `LazyGeometry(base_kind="shape", base_payload=(impl, params))` を返す（`_lazy_shape`）。
    - `ShapesAPI.label(self, uid: str) -> ShapesAPI` は「次の shape 呼び出しに対するラベル」を `_ShapeCallContext` に格納するだけ。
    - `_build_shape_method()` 内で `get_active_runtime()` を取得し、`runtime.before_shape_call(...)` と `runtime.relabel_shape(shape_name, ctx_label)` を呼び出す。
  - `LazyGeometry` (`src/engine/core/lazy_geometry.py`)
    - `base_kind: Literal["shape", "geometry"]` と `base_payload` を持つ。
    - shape の場合の `base_payload` は `(shape_impl, params_dict)`。
    - `realize()` では `(shape_impl, params)` から Geometry を生成し、effect plan を適用する。
    - 便宜メソッドとして `translate/scale/rotate/__add__` はあるが、`label()` は未実装。
  - Parameter GUI (`src/engine/ui/parameters/runtime.py`, `value_resolver.py`)
    - `ParameterRuntime.before_shape_call()` で `ParameterContext(scope="shape", name=shape_name, index=index)` を構築し、Descriptor を登録。
    - `ParameterContext.category` は shape のとき `name` / `name_1` / `name_2` ... の規則でヘッダ名を決定（shape ヘッダ分割は実装済み）。
    - `ParameterRuntime.relabel_shape(shape_name, base_label)` は、`shape_name` ごとのカテゴリラベルを `_shape_label_by_name` に保持し、既存 Descriptor の `category` をまとめて書き換える。
      - 同じ `shape_name` に対して複数回 `relabel_shape` を呼んでも、現状は 1 つの `display_label` のみを保持（呼び出しごとの連番までは付けていない）。
- Effects 側のラベル
  - `PipelineBuilder.label(self, uid: str)` は RuntimeAdapter を通じて `ParameterRuntime` に `pipeline_label` を伝える。
  - `E.pipeline` / `E.label("uid")` / `E.<effect>(...)` いずれも「チェーンの任意位置」で `.label()` を呼び出せる設計になっている。
- 型スタブ (`src/api/__init__.pyi`, `tools/gen_g_stubs.py`)
  - `_GShapes` Protocol の各メソッドは `-> Geometry` を返す形になっており、`LazyGeometry` は表に出ていない。
  - `Geometry` には `label()` メソッドは定義されておらず、`G.polygon().label()` のようなコードは「`Geometry` に `label` が無い」という警告になる。
  - `tools/gen_g_stubs.py` では shape 側の戻り値を一律で `Geometry` にしている。
- ドキュメント
  - `architecture.md` では「`G.<name>(...)` は既定で `LazyGeometry`（spec）を返す」と説明されており、実装と一致している。
  - `parameter_gui_shape_header_split_plan.md` の 5 節で「`G.label("title").text(...)` などのチェーン」「shape 呼び出しごとにヘッダ分割・ラベル反映」が仕様として整理されているが、`G.text(...).label("title")` までは実装されていない。

## 課題（問題点の整理）

- ランタイム挙動
  - `G.label("title").text(...)` は動作するが、`G.text(...).label("title")` / `G.polygon(...).label("title")` は `LazyGeometry`（実際の戻り値）に `label` が無いため実行時エラーになる。
  - `ParameterRuntime.relabel_shape` は `shape_name` 単位でカテゴリを書き換えるため、「同じ shape を複数回呼び出したときのラベル連番（`title`, `title_1`, ...）」の仕様が現状コードに完全には反映されていない可能性がある。
- 型チェック・開発体験
  - `_GShapes` が `-> Geometry` を返すことにより、エディタ/型チェッカでは `G.polygon().label()` が「`Geometry` に `label` が無い」と警告される。
  - 実体は `LazyGeometry` であるため、型情報と実装の乖離が大きく、`label` 以外の Lazy ベースの API 拡張にも影響する。
- 設計整合性
  - Shapes 側では「`G` にだけ `label()` がある」状態で、LazyGeometry には無い。一方 Effects 側は `PipelineBuilder` 自身が `.label()` を持つ。
  - `parameter_gui_shape_header_split_plan.md` の「チェーン任意位置」という仕様と、実装/型情報が噛み合っていない。

## 方針（ざっくりした解決イメージ）

- Shapes 側も「LazyGeometry に `label()` を持たせる」方向で、Effects の `PipelineBuilder` と揃える。
  - `G.<shape>(...)` の戻り値（LazyGeometry）に対して `.label("uid")` を呼ぶと、Parameter GUI の shape カテゴリにラベルを適用する。
  - `G.label("uid").<shape>(...)` という既存の呼び出しは残しつつ、内部的には LazyGeometry 側の `label()` に寄せる実装も検討する。
- `LazyGeometry.label()` の実装は、必要最小限の情報だけを Runtime に伝える。
  - `get_active_runtime()` で ParameterRuntime を取得し、base が shape のときのみ動作させる（geometry ベースの LazyGeometry では no-op）。
  - shape 名の特定は、`base_payload` 内の `shape_impl` と `shapes.registry.get_registry()` の値を突き合わせて行う（`impl = getattr(fn, "__shape_impl__", fn)` で正規化して比較）。
  - 見つかった shape 名に対して `runtime.relabel_shape(shape_name, uid)` を呼び、既存のラベルロジックを再利用する。
- 型スタブは段階的に整理する。
  - 最小案: `_GShapes` の戻り値型は現状のまま `Geometry` としつつ、`Geometry` 側に `def label(self, uid: str) -> Geometry: ...` を追加して警告のみ解消する（実体は LazyGeometry なので多少の型の嘘は許容）。
  - 改善案: `_GShapes` の戻り値を `Geometry | LazyGeometry` または `LazyGeometry` に寄せ、`LazyGeometry` 型を `api` 層からも参照できるようにする（Breaking だが、まだ非公開リポのため許容可能）。
  - どこまで型の厳密さを追うかは、このファイルでユーザーとすり合わせる。

## やること（チェックリスト）

### 1. 仕様の確定・確認事項

- [ ] `G.text(...).label("title")` のように「shape 呼び出し後に label」を掛ける書き方を正式にサポートしてよいか確認する。
- [ ] ラベル適用の単位を「shape 名」単位にするか「shape 呼び出しインスタンス」単位にするかを決める。
  - 例: `G.text(...).label("title")` を 2 回呼んだとき、ヘッダ名を `title` / `title_1` とするか、どちらも `title` にするか。
- [ ] `LazyGeometry` 以外（`Geometry` や `Layer`）に対して `.label()` を許容するかどうか（no-op か、エラーとするか）を決める。
- [ ] 型スタブの目標粒度（`Geometry` に嘘の `label` を載せるか、`LazyGeometry` を正式に API として出すか）を決める。

### 2. ランタイム実装設計（LazyGeometry.label）

- [ ] `LazyGeometry` に `def label(self, uid: str) -> "LazyGeometry"` を追加する設計をまとめる。
  - base が `"shape"` のときのみラベル処理を行い、それ以外は self をそのまま返す。
  - `uid` は `str(uid).strip()` で正規化し、空文字列は無視（no-op）とする。
- [ ] `LazyGeometry` から shape 名を特定するヘルパを設計する。
  - `shapes.registry.get_registry()` を使い、登録済み shape 関数から name → fn を取得。
  - 各 `fn` に対して `impl = getattr(fn, "__shape_impl__", fn)` を取り、`impl is shape_impl` で一致を取る。
  - 一致する name が複数見つかった場合は、最初の 1 件のみ採用（想定外ケースなのでログ/コメントで軽く言及）。
- [ ] `get_active_runtime()` が存在しない（Parameter GUI 無効）場合や、shape 名が解決できなかった場合の挙動を決める（単に self を返すだけ）。
- [ ] 既存の `ShapesAPI.label()` との重複を整理する。
  - `G.label("uid").text(...)` では、現在 `_ShapeCallContext` を経由して `runtime.relabel_shape(shape_name, uid)` を shape 呼び出し直前に行っている。
  - 将来的に `ShapesAPI.label()` の実装を「`G.label("uid")` → 次の LazyGeometry に `.label("uid")` を適用する薄いラッパ」に寄せるかどうかを検討する（互換性とシンプルさのトレードオフ）。

### 3. ParameterRuntime / ラベルロジックの整合確認

- [ ] `ParameterRuntime.relabel_shape(shape_name, base_label)` の現行挙動を再確認し、shape ヘッダ分割仕様（`text`, `text_1`, ...）と矛盾しないかチェックする。
  - 特に「同じ shape 名に対して複数回ラベルを適用したとき」に、期待どおり `title`, `title_1`, `title_2` となるかを確認する。
  - 必要なら `_assign_shape_label` の key を `shape_name` ではなくラベル文字列ベースにするなどの変更案を検討する。
- [ ] `LazyGeometry.label()` から `relabel_shape` を呼び出したときに、すでに登録済みの Descriptor だけが更新されることを確認する（未登録 shape には影響しないこと）。
- [ ] `parameter_gui_shape_header_split_plan.md` で定義済みの仕様と、新しい `label` 挙動の整合性をチェックし、必要なら同ファイルの仕様文言をアップデートする。

### 4. 型スタブ・スタブ生成スクリプトの更新

- [ ] `src/api/__init__.pyi` の `_GShapes` 定義を見直し、戻り値型と `label` 呼び出しの型整合性を取る案を 2 パターン用意する。
  - 案 A（最小変更）: 既存どおり戻り値は `Geometry` のままとし、`Geometry` に `def label(self, uid: str) -> Geometry: ...` を追加して警告だけ解消する。
  - 案 B（精度重視）: 戻り値を `LazyGeometry`（または `Geometry | LazyGeometry`）に変更し、`LazyGeometry` 型を `api` からも import 可能にする。
- [ ] `tools/gen_g_stubs.py` に `LazyGeometry` を import させるかどうか、どの層まで型を露出させるかを決める。
- [ ] 選択した案に応じてスタブ生成ロジックを修正し、`python -m tools.gen_g_stubs` の出力とテスト（`tests/stubs/test_g_stub_sync.py`）が通ることを確認する。

### 5. テスト追加・更新

- [ ] `tests/ui/parameters` 配下に「shape ラベル適用」用のテストを追加する。
  - `G.text(...).label("title")` 相当のコードパスで、Parameter GUI 上のカテゴリ名が期待どおりになるかを検証する（ユニットテストベースでよい）。
  - `G.label("title").text(...)` との挙動差/互換性も確認する。
- [ ] `LazyGeometry.label()` 単体の挙動テストを追加する。
  - shape ベース / geometry ベース / Runtime 無効 / shape 名解決失敗 の各ケースで no-op かラベル適用かを確認する。
- [ ] 型レベルの検証（mypy あるいは pyright 相当の simple スクリプト）で、`G.polygon().label("uid")` に警告が出ないことを確認する（必要なら `tests/stubs` に近い場所でスモークテスト化）。

### 6. ドキュメント更新

- [ ] `architecture.md` の Shapes/Parameter GUI セクションに「`G.<shape>(...).label("title")` もサポートする」旨を追記し、Examples に 1 つ追加する。
- [ ] `parameter_gui_shape_header_split_plan.md` の 5 節（G.label API 拡張）に、最終的な挙動（どのパターンで label が効くか、ヘッダ名との対応）を反映させる。
- [ ] ルート `AGENTS.md` の Parameter GUI 関連記述に、`G.label()` のチェーン位置について 1 行程度の補足を入れるか検討する。

### 7. 移行・互換性の確認

- [ ] 既存のスケッチ（`main.py` や `sketch/` 配下）で `G.label()` を利用している箇所を洗い出し、新挙動で問題が出ないか目視確認する。
- [ ] もし `_GShapes` の戻り値を `LazyGeometry` に変更する場合、外部コード（あれば）の型エラーや挙動への影響を整理し、このリポがまだ配布前であることを前提に許容範囲かどうか判断する。

---

### ユーザーとの合意事項（確認済み）

1. `G.text(...).label("title")` / `G.polygon(...).label("title")` のような「shape 呼び出し後の label」を正式にサポートする方向で進める。
2. ラベルの連番仕様:
   - 同じ shape 名 + 同じラベル文字列で複数回呼んだ場合、ヘッダは `title`, `title_1`, `title_2` ... のように分割する（1 つにまとめない）。
3. 型スタブの方針:
   - まずは最小変更案で進め、`Geometry` にも `label()` を追加する形で警告解消を優先する（`LazyGeometry` の型公開は後続の検討事項とする）。

これらの前提に基づき、このチェックリストに沿って実装・テスト・ドキュメント更新を進め、完了した項目にチェックを付けていきます。
