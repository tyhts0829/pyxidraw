# E.label() 後置 label 対応 実装計画（案）

目的:  
`E.label(uid)` のラベル指定が「E の直後」だけでなく、`E.affine().fill().label(uid="...")` のように **effect チェーンの後ろに付けた場合でも Parameter GUI 上で有効になる** ようにする。あわせて、`E.pipeline.label(uid)` / `PipelineBuilder.label(uid)` との挙動を整理し、「label はパイプライン全体の表示ラベル」を一貫して表すようにする。

前提・現状把握（確認用）:

- 現状 API/実装
  - `src/api/effects.py`
    - `_EffectsAPI.label(uid)` は `PipelineBuilder()` を生成してから `builder.label(uid)` を呼ぶ（`E.label(uid).affine().fill()` 用）。
    - `PipelineBuilder.label(uid)` は `self._label_display` を更新するだけで、すでに積まれている steps には影響しない。
    - `PipelineBuilder.__getattr__` 内で `get_active_runtime()` から `ParameterRuntime` を取得し、`before_effect_call(...)` を呼んだ結果を使って `params_signature` を計算し、`self._steps` に積む。
    - パイプライン UID (`self._uid`) は **最初の effect 追加時** に `runtime.next_pipeline_uid()` から発行される。
  - `src/engine/ui/parameters/runtime.py`
    - `ParameterRuntime.before_effect_call(...)` は `pipeline_uid` と `pipeline_label`（= `PipelineBuilder._label_display`）を受け取り、`ParameterContext` を生成して `ParameterValueResolver.resolve(...)` を呼ぶ。
    - `pipeline_label` が非空の場合、`_pipeline_label_by_uid` / `_pipeline_label_counter` を使ってフレーム内インスタンスごとの `display_label` を決め、`context.pipeline_label` に格納する。
  - `src/engine/ui/parameters/value_resolver.py`
    - `ParameterContext.descriptor_prefix` は effect の場合 `effect@{pipeline_uid}.{name}#{index}` 形式（`pipeline_uid` が空のときは `"effect.{name}#{index}"`）。
    - Descriptor の `category` は `context.pipeline_label or context.pipeline or context.scope` で決まる（effect は `category_kind="pipeline"`）。
    - Descriptor の `pipeline_uid` / `step_index` も `context` から埋められる。
  - `src/engine/ui/parameters/snapshot.py`
    - `SnapshotRuntime.before_effect_call(...)` も `pipeline_uid` を使って prefix `effect@{pipeline_uid}.{effect_name}#{step_index}` を組み立て、`extract_overrides(...)` から受け取った override を適用する。
    - `pipeline_label` 引数は現状未使用。
- 問題点
  - `E.label(uid="...").affine().fill()` のように **最初に label を付ける** ケースでは、`before_effect_call` 呼び出し時に `pipeline_label` が設定されているので、Parameter GUI 上のカテゴリが期待どおりラベル名（＋連番）になる。
  - 一方 `E.affine().fill().label(uid="...")` のように **effect の後ろで label を呼ぶ** と、`label()` 実行時にはすでに全ステップが `before_effect_call` 済みであり、`pipeline_label` を使ったカテゴリ/Descriptor ID は確定済みのため、後からの `label()` は GUI に反映されない。
  - 現状の `PipelineBuilder.label` は「これから追加されるステップのラベル」を指定する設計になっており、「パイプライン全体のラベルを後から付ける／変更する」用途を満たせていない。

---

## A. 目標仕様（E.label/PipelineBuilder.label の挙動整理）

- [x] label の意味を「パイプライン全体の表示ラベル」として明示する。
  - [x] `E.label(uid)` / `E.pipeline.label(uid)` / `PipelineBuilder.label(uid)` はすべて同じ概念（pipeline display label）を指す。
  - [x] label は **チェーン中のどの位置で呼んでも最終的にパイプライン全体に適用される**（後置 `.label` を許可）。
- [x] GUI 観点での期待挙動を定義する。
  - [x] Parameter GUI のカテゴリ見出しは `pipeline_label`（＋フレーム内連番）を最優先で使う。
  - [x] `E.affine().fill().label(uid="poly_effect")` でも、`E.label(uid="poly_effect").affine().fill()` と同じカテゴリ名になる。
  - [x] ID 形式（`effect@...`）やキャッシュキーは引き続き `pipeline_uid` を主キーとし、label の変更は **UI 表示と override の紐付けのみに影響**させる（動作・キャッシュには影響させない）。
- [x] 複数回の label 呼び出し時の扱いを決める。
  - [x] 単純に「**最後に呼ばれた label が有効**」とする（前置・途中・後置を問わない）。
  - [x] label 未指定の場合は従来どおり `pipeline_uid` か `scope` 名でカテゴリを構成する。

---

## B. 実装方針の大枠

- [x] `PipelineBuilder.label()` を「これから追加されるステップのラベル指定」から「パイプライン全体のラベル更新」に拡張する。
  - [x] `self._label_display` の更新は維持しつつ、「すでに UID が振られているパイプライン」については ParameterRuntime/ParameterStore にもラベル変更を伝える。
- [x] Parameter GUI 側では、「パイプライン UID → 表示ラベル」の対応を Runtime が一元管理し、Descriptor のカテゴリ名を必要に応じて更新できるようにする。
  - [x] 新規フレーム開始時 (`begin_frame`) に `_pipeline_label_by_uid` を初期化する現在の設計は維持。
  - [x] label を後から設定した場合でも、同一フレーム中に作られた Descriptor のカテゴリを「ラベルベース」に揃える。
- [x] SnapshotRuntime 側は prefix/ID 形式を現状維持し、`pipeline_uid` ベースの override 適用を続ける。
  - [x] この改善では ID 形式自体は変えず、「カテゴリの表示（ヘッダ名）」と「override の prefix（effect@...）」を切り分ける。

---

## C. 具体タスク（チェックリスト）

### C-1. 現状確認とスコープ固定

- [x] `docs/spec/pipeline.md` / `architecture.md` の label 説明を読み直し、「label は表示ラベルであり内部 UID には影響しない」という前提を再確認する。
- [x] `sketch/251118.py` / `sketch/251117.py` / `sketch/251115.py` / `sketch/251113.py` の label 使用パターンを洗い直し、後方互換で壊したくないケースを列挙する。
- [x] 既存の `sketch_251118_lazy_loop_fix_plan.md` との整合を確認し、「ラベル付きパイプラインを 1 つの論理パイプラインとして扱う」方針と矛盾しないことを確認する。

### C-2. ParameterRuntime に「パイプライン再ラベル」API を追加

- [x] `src/engine/ui/parameters/runtime.py` に、パイプライン UID とベースラベルから表示ラベルを再決定するヘルパを追加する。
  - [x] 関数案: `def _assign_pipeline_label(self, pipeline_uid: str, base_label: str) -> str: ...`
    - [x] `before_effect_call` 内のラベル決定ロジック（`_pipeline_label_by_uid` / `_pipeline_label_counter` の扱い）をこのヘルパに移す。
    - [x] 返り値として「表示ラベル」（例: `"poly_effect_1"`）を返す。
- [x] 上記ヘルパを利用する公開メソッドを追加する。
  - [x] 関数案: `def relabel_pipeline(self, pipeline_uid: str, base_label: str) -> None: ...`
    - [x] `base_label` が空/無効な場合は何もしない。
    - [x] `_assign_pipeline_label(...)` を呼び出して `_pipeline_label_by_uid` / `_pipeline_label_counter` を更新する。
    - [x] 対象 `pipeline_uid` を持つ effect 用 Descriptor の `category` を「新しい表示ラベル」に揃えるよう、`ParameterStore` と協調する。
- [x] 既存の `before_effect_call` 内では、直接 `_pipeline_label_by_uid` を触らず `_assign_pipeline_label` を使うように書き換える。

### C-3. ParameterStore/Descriptor メタの更新手段を用意

- [x] `ParameterStore` に「既存 Descriptor の一部フィールドを書き換える」ための最小 API を追加する。
  - [x] 例: `def update_descriptors(self, fn: Callable[[ParameterDescriptor], ParameterDescriptor]) -> None: ...`
    - [x] 内部で `_descriptors` を走査し、fn(desc) が同一 ID の別インスタンスを返した場合に差し替える。
    - [x] `id` は変えない前提とし、`category` や `category_kind` など UI 用フィールドのみ変更対象とする。
- [x] `ParameterRuntime.relabel_pipeline(...)` から上記 API を使い、`desc.pipeline_uid == pipeline_uid` を満たす effect Descriptor の `category` を「表示ラベル」に更新する。
  - [x] 形状 (`source="shape"`) の Descriptor は対象外とする。
  - [x] `category_kind` は既存の `"pipeline"` を維持。

### C-4. PipelineBuilder.label() から Runtime への連携

- [x] `src/api/effects.py:PipelineBuilder.label` を拡張し、「すでに UID が振られている場合」に Runtime へ relabel を伝える。
  - [x] `self._label_display` の更新はこれまでどおり行う。
  - [x] `runtime = get_active_runtime()` を取得し、`runtime is not None` かつ `self._uid` が `None` でない場合にだけ `runtime.relabel_pipeline(pipeline_uid=str(self._uid), base_label=text)` を呼ぶ。
  - [x] Runtime が取得できない場合（Parameter GUI 無効）は何もしない（既存挙動と同様に label を「将来のためのメモ」として扱う）。
- [x] `PipelineBuilder.__getattr__` 内での `before_effect_call` 呼び出し部分を `_assign_pipeline_label` ベースに書き換えることで、「前置 label」と「後置 label」でラベルの決定ロジックを共通化する。
  - [x] `self._uid` が未設定の場合のみ UID を発行する既存ロジックは維持。

### C-5. SnapshotRuntime との整合確認

- [x] `src/engine/ui/parameters/snapshot.py:SnapshotRuntime.before_effect_call` を確認し、このタスクでは ID 形式を変更しないことを明示する。
  - [x] prefix は引き続き `effect@{pipeline_uid}.{effect_name}#{step_index}` を用いる。
  - [x] `pipeline_label` 引数は今後の拡張余地として残し、今回は利用しない。
- [x] `extract_overrides` → `SnapshotRuntime.before_effect_call` の流れが、今回の「カテゴリ名の変更」によって壊れないことを確認する（override のキーは Descriptor ID ベースであり、`id` を変更しない限り影響しない想定）。

### C-6. 動作確認・テスト

- [x] 単体テスト（新規）
  - [x] `tests/ui/parameters` か新規 test モジュールで、`ParameterRuntime` / `ParameterStore` を直接使うテストを追加する。
    - [x] ケース 1: `E.label(uid="polygons").affine().fill()` パターンでカテゴリが `"polygons_1"` のようになることを確認。
    - [x] ケース 2: `E.affine().fill().label(uid="poly_effect")` パターンで、effect Descriptor のカテゴリが `"poly_effect_1"` になることを確認。
    - [x] ケース 3: 同一フレーム内で同じ label のパイプラインを複数作ったとき、`poly_effect_1`, `poly_effect_2`, ... と連番が付くことを確認。
  - [x] 可能であれば SnapshotRuntime を使ったテストも追加し、override の適用に問題がないことを確認する（ID 形式が変わらないことを前提）。
- [ ] 簡易動作確認（手動想定）
  - [ ] `sketch/251118.py` を `use_parameter_gui=True` で実行し、`E.affine().fill().label(...)` のパイプラインが GUI 上で期待どおりのカテゴリ名になることを確認。
  - [ ] `sketch/251117.py` など既存の label 使用スケッチでカテゴリ名・挙動に regress がないことを確認。

### C-7. ドキュメント更新

- [x] `docs/spec/pipeline.md` の label 説明に、「label はチェーンの前後どこで呼んでもパイプライン全体に適用される」旨を追記する。
- [x] `architecture.md` の Parameter GUI / Pipeline セクションに、「カテゴリ名は pipeline_label（存在しない場合は pipeline_uid など）から決まる」こと、および「後置 label でも有効である」ことを簡潔に反映させる。
- [ ] 必要に応じて `sketch_251118_lazy_loop_investigation.md` から本計画ファイルへの言及を追加する。

---

## D. 事前に確認したい点・オープンな判断

- [x] **ID 形式を label ベースに拡張するかどうか**  
      現計画では「カテゴリ名のみ label ベース」にし、Descriptor ID / override prefix は従来どおり `pipeline_uid` ベースのままとする。 ；はい
  - メリット: 既存の overrides / SnapshotRuntime ロジックへの影響が最小。
  - デメリット: 将来的に「ラベルを完全な主キーにしたい」場合には、別途 ID 形式の変更（`sketch_251118_lazy_loop_fix_plan.md` 側の案）を実装する必要がある。
- [x] **後置 label の優先順位**  
      複数回 label を呼んだ場合に「最後の label を優先する」仕様で問題ないか。 ；はい
      例: `E.label("A").affine().label("B").fill().label("C")` → 最終的なカテゴリは `"C_*"` になる。
- [x] **既存の overrides の扱い**  
      この変更では Descriptor ID を変えない前提だが、「カテゴリ名が変わることでユーザーが保存済み overrides を見失う」ことをどこまで許容するか（現状ユーザーは居ない前提なので、大きな問題ではなさそう）。；問題なし

---

このチェックリストと方針で問題なければ、この計画に沿って `PipelineBuilder.label` / `ParameterRuntime` / `ParameterStore` を中心に実装し、進行に応じて完了済み項目にチェックを付けていく。必要に応じて、ID 形式の label ベース化など、追加の改善案も別途計画ファイルとして分離する。\*\*\*
