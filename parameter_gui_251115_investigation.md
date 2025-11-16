## 対象バグ

- 対象スケッチ: `sketch/251115.py`
- 現象: parameter_gui で polygon のみ Shape ヘッダーとして表示され、text や line が Effect ヘッダー側にまとまってしまう。

## 原因の整理

1. **カテゴリ表示ロジック（現状）**
   - GUI 側のカテゴリヘッダー種別は `src/engine/ui/parameters/dpg_window.py:686` 付近の `_build_grouped_table`→`flush()`→`_category_kind()` によって決まる。
   - `_category_kind()` は **カテゴリ内の descriptor の `source` を見て決めている**:
     - `any(it.source == "effect" for it in items)` が真なら `"pipeline"`（Effect 用テーマ）
     - そうでなければ `"shape"`（Shape 用テーマ）
   - つまり、「そのカテゴリに effect 由来のパラメータが 1 つでも含まれていると、そのカテゴリヘッダーは pipeline（Effect）側のテーマで描画される」設計になっている。

2. **パラメータ descriptor の `source` と `category`**
   - descriptor の生成は `src/engine/ui/parameters/value_resolver.py` の `ParameterValueResolver` で行われる。
   - `ParameterContext.scope` に応じて `source` と `category` がセットされる:
     - `scope="shape"` のとき: `source="shape"`, `category=context.name`（= 形状名）
     - `scope="effect"` のとき: `source="effect"`, `category=context.pipeline_label or context.pipeline or context.scope`
   - `ParameterContext` は `src/engine/ui/parameters/value_resolver.py:41` で定義され、`scope` は `ParameterRuntime` から渡される。

3. **ParameterRuntime 側の scope 設定**
   - `src/engine/ui/parameters/runtime.py`:
     - 形状呼び出し前: `before_shape_call()` → `ParameterContext(scope="shape", name=shape_name, index=index)`
     - エフェクト呼び出し前: `before_effect_call()` → `ParameterContext(scope="effect", ...)`
   - したがって、**text や line などの shape パラメータも本来 `source="shape"` で登録される**。

4. **sketch/251115.py におけるカテゴリ構成の推測**
   - `sketch/251115.py` の構造:
     - polygon: `G.polygon()` → shape パラメータ（polygon の n_sides など）→ `category="polygon"` で `source="shape"`.
     - text: `G.text()` → text パラメータ（`text.__param_meta__` 由来）→ `category="text"` で `source="shape"`.
     - line: `G.line()` → line パラメータ（`line.__param_meta__` 由来）→ `category="line"` で `source="shape"`.
     - effects: `E.pipeline.affine().fill().subdivide().wobble().dash(...).offset().fill()` など → それぞれ `source="effect"` で、`category` は pipeline ラベルまたは UID。
   - renderer の grouping は「descriptor.category ごと」に行うため、polygon/text/line はそれぞれ別カテゴリ（"polygon"/"text"/"line"）としてグループ化される。

5. **polygon だけ Shape ヘッダーに見える理由（仮説）**
   - `_category_kind()` は「カテゴリ内に effect が含まれるかどうか」でヘッダー種別を決定するため、**effect と shape が同じカテゴリ名を共有すると、そのカテゴリは Effect ヘッダーになります**。
   - 現状のロジックでは:
     - 形状カテゴリ: `"polygon"`, `"text"`, `"line"`（いずれも `source="shape"`）
     - パイプラインカテゴリ: `pipeline_label` が未指定なら `"p0"`, `"p1"` ... のような UID か `"effect"` 系。
   - ここから、「polygon カテゴリには effect 由来の descriptor が混ざっておらず」「text/line カテゴリにはなんらかの effect 由来 descriptor が同じ category 名で混入している」ことが示唆される。
     - 例えば、`ParameterContext` における `pipeline_label` の扱いと、`value_resolver` における `category` の決め方が特定条件で text/line と同じ文字列になるケースがありうる。

6. **根本的な問題点（設計）**
   - `_category_kind()` の判断基準が「カテゴリ内の descriptor の `source` の混在有無」に依存しており、**カテゴリの意味（Shapeカテゴリか Effectカテゴリか）を明示的に持っていない**。
   - その結果、以下のような設計上の脆さがある:
     - shape/effect どちらでも起こりうる **カテゴリ名の衝突** によって、本来 Shape ヘッダーであるべきグループが Effect ヘッダーとして描画され得る。
     - 同じカテゴリ名の中に `source="shape"` と `source="effect"` が混在した場合、カテゴリヘッダーは無条件に Effect 扱いになる（`any(it.source == "effect")`）。
   - polygon だけが Shape ヘッダーになり、text/line が Effect ヘッダーになるのは、**polygon カテゴリには effect 由来 descriptor が混ざっておらず、text/line カテゴリには混ざっている**ためと考えられる。

## 採用方針（A案）と設計の方向性

### 方針A: カテゴリ種別を Descriptor に明示的に持たせる（採用）

- ゴール:
  - 「そのカテゴリが Shape 用か Pipeline(Effect) 用か」を **Descriptor 自身が明示的に持つ** ようにし、ヘッダー描画ロジックはそれを参照するだけにする。
  - shape/effect のカテゴリ名衝突に左右されない設計にする。

- 設計方針（概要）:
  - `src/engine/ui/parameters/state.py` の `ParameterDescriptor` に、カテゴリ種別フィールドを追加する。
    - 例: `category_kind: Literal["shape", "pipeline", "hud", "display"] = "shape"`
    - 既定値は `"shape"` とし、HUD/Display/pipeline 系で明示的に上書きする。
  - Descriptor 生成箇所で `category_kind` を一意に決める:
    - `ParameterValueResolver` 内:
      - `context.scope == "shape"` のとき: `category_kind="shape"`
      - `context.scope == "effect"` のとき: `category_kind="pipeline"`
    - `ParameterManager` 内で手動生成している runner/background/HUD 用 descriptor:
      - 背景・線色など描画設定: `category_kind="display"`（または `"runner"`）
      - HUD 関連: `category_kind="hud"`
  - `_category_kind()` の責務を縮小:
    - `items` が空でない前提で `items[0].category_kind` を返す単純な実装にする。
    - カテゴリ内で `category_kind` が混在している場合は、ログ or assert で検知（将来のバグ早期発見用）。
  - `ParameterThemeConfig.categories` との対応:
    - キーを `category_kind` ベース（"shape"/"pipeline"/"hud"/"display" 等）に整理する。
    - 既存の `"shape"` / `"pipeline"` キーはそのまま活かしつつ、HUD/Display 用キーを明示する。

- メリット:
  - ロジックが単純かつ意図が明確になり、shape/effect の混在やカテゴリ名衝突に強くなる。
  - HUD/Display など、UI カテゴリ拡張時の影響範囲が読みやすくなる。

- デメリット・注意点:
  - `ParameterDescriptor` のフィールド追加に伴い、型・スタブ・テスト・ドキュメントの更新が必要。
  - `ParameterDescriptor` を直接生成している既存コード（ParameterManager など）をすべて洗い出して追従させる必要がある。

### （参考）他案について

- 方針B（カテゴリ名 prefix）/方針C（source 優先ルール変更）は今回採用しない。
  - どちらも「カテゴリの意味を暗黙に推論する」設計であり、今回のようなバグ再発リスクが残る。

## 実装改善チェックリスト（方針Aベース）

**0. 事前準備**
- [ ] `engine/ui/parameters` 配下の AGENTS.md・architecture.md を再確認し、カテゴリ/テーマ設計に関する既存の意図を洗い直す。
- [ ] `tests/ui/parameters` の既存テストを読み、カテゴリ表示やヘッダ種別に関する前提がないか確認する。

**1. `ParameterDescriptor` への `category_kind` 追加**
- [ ] `src/engine/ui/parameters/state.py` の `ParameterDescriptor` に `category_kind` フィールドを追加し、型を `Literal["shape", "pipeline", "hud", "display"]`（仮）とする。
- [ ] 既定値を `"shape"` に設定し、既存コードからの生成で明示指定がない場合は Shape 扱いにする。
- [ ] 追加フィールドに関する docstring を更新し、「ヘッダーのテーマ決定に使用するカテゴリ種別」であることを明記する。

**2. Descriptor 生成箇所の更新**
- [ ] `src/engine/ui/parameters/value_resolver.py` で Descriptor を生成しているすべての箇所に `category_kind` を追加する。
  - [ ] shape 由来 (`context.scope == "shape"`) の Descriptor で `category_kind="shape"` をセット。
  - [ ] effect 由来 (`context.scope == "effect"`) の Descriptor で `category_kind="pipeline"` をセット。
- [ ] `src/engine/ui/parameters/manager.py` で手動生成している Descriptor（background/line_color/HUD 等）に対して、用途に応じた `category_kind` を設定する。
  - [ ] 背景・線色など描画設定 → `category_kind="display"`（または `"runner"` として統一）。
  - [ ] HUD 関連 → `category_kind="hud"`.

**3. `_category_kind()` の単純化と一貫性チェック**
- [ ] `src/engine/ui/parameters/dpg_window.py` の `_category_kind()` を、「`items` 先頭要素の `category_kind` を返す」実装に差し替える。
- [ ] `items` 内で `category_kind` が混在している場合に検知できるよう、ログ or `assert` を追加するかどうかを検討し、最小限の検査を入れる。
- [ ] `_build_grouped_table()` → `flush()` → `_category_kind()` の呼び出し経路で、`items` が空で呼ばれないことを確認し、必要に応じて防御コードを追加する。

**4. テーマ設定との整合性の見直し**
- [ ] `src/engine/ui/parameters/state.py` の `ParameterThemeConfig.categories` の docstring を更新し、「キーが `category_kind` と対応する」前提を明文化する。
- [ ] `src/engine/ui/parameters/dpg_window.py` の `_get_category_header_theme()` / `_get_category_table_theme()` で参照しているカテゴリ名と、`category_kind` の値の対応を確認する。
- [ ] HUD/Display 用のテーマキー（"HUD"/"Display" など）が、`category_kind` の値と衝突していないか整理し、必要ならリネーム or マッピングレイヤを導入する。

**5. 動作確認とテスト**
- [ ] `sketch/251115.py` を実行し、polygon/text/line のカテゴリヘッダーがすべて Shape テーマ（`category_kind="shape"`）として描画されることを目視確認する。
- [ ] 他の代表的なスケッチ（Shapes 重視・Effects 重視・HUD 利用など）でも、ヘッダー種別が期待どおりか確認する。
- [ ] `tests/ui/parameters` に、`category_kind` を前提としたテストを追加/更新する。
  - [ ] `ParameterDescriptor` の新フィールドに関するシリアライズ/復元/永続化周りのテストが必要か確認する。
  - [ ] `_category_kind()` の挙動が `category_kind` にのみ依存することを検証するテストを追加する。

**6. スタブ・型・ドキュメントの更新**
- [ ] `src/api/__init__.pyi` 等、`ParameterDescriptor` を公開しているスタブに新フィールドを反映させる（必要であれば）。
- [ ] `python -m tools.gen_g_stubs` によるスタブ再生成が必要かを確認し、必要なら実行して差分を確認する。
- [ ] `architecture.md` に parameter_gui のカテゴリ種別とテーマ適用ルールを追記し、今回の設計変更を反映する。
- [ ] `src/engine/ui/parameters/AGENTS.md` に「category_kind と Theme の対応」「カテゴリ名と表示ラベルのルール」を簡潔に追記する。

**7. 追加で検討したい点（要相談）**
- [ ] text/line など、複数 shape が混在するスケッチにおいて、「Shape 全体のグローバルカテゴリ」と「個別 shape ごとのカテゴリ」をどう扱うべきか（ヘッダ階層を増やすか、現状のまま平坦にするか）。
- [ ] effect 側も pipeline 単位だけでなく、effect 名単位でカテゴリを切るかどうか（現在は `pipeline_label`/UID ベース）。
- [ ] HUD/Display のようなランタイム/描画関連パラメータを、Shape/Effect と並列に見せるか、別タブ/セクションに分離するか。

---

上記チェックリストに対して、どの範囲までを今回の修正対象とするかを決めていただければ、その方針に沿って実際のコード修正とテスト追加を進めます。
