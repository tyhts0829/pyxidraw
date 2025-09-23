# ユーザー非認知のキャッシュ運用（Effects/Shapes）仕様（更新提案）

目的
- ユーザーにキャッシュや build の存在を意識させず、CC/GUI/時間`t`を含む通常操作で「有効パラメータが変わらないフレームは再計算しない」を実現する。
- GUI 有効時でも Shape 生成にキャッシュを効かせ、全体のレイテンシとスループットを改善する。
- t/cc/gui はいずれも「shape/pipeline への入力パラメータ」とみなし、同一モデルで取り扱う。

スコープ（対象と非対象）
- 対象: `api.effects` の Pipeline 実行、`api.shapes` の生成、`engine.ui.parameters`（Runtime 経由のパラメータ適用）。
- 非対象: 外部ファイル入出力・レンダラ・MIDI 実装詳細（CC 値供給は既存の snapshot に依存）。

達成したい UX（ユーザー視点の要件）
- ユーザーは build やキャッシュ戦略を呼び分けない。常に同じ書き方でよい。
- CC/GUI/時間`t`のいずれかで「有効パラメータが変わった」フレームだけ再計算。変わらなければ全ステップをスキップ（即ヒット）。
- GUI を ON にしても Shape 側の LRU が効く（＝生成済み形状の再利用で安定レイテンシ）。
- 丸め（量子化）やヒステリシス等の安定化は裏側で自動適用。ユーザーはしきい値を意識しない。

動作イメージ（main2.py を想定）
```python
from api import G, E

# 例: CC/GUI/t に応じて affine を操作、ユーザーは build を書かない
pipe = (E.pipeline
          .affine()  # params は静的既定値でも良い（実行時に解決）
          .extrude(distance=10, direction=(0,0,1)))  # build は書かない

def draw(t, cc):
    g = G.polyhedron().scale(200).translate(200, 200, 0)
    return pipe(g)  # ← 内部で入力スナップショット(t/cc/gui)を統合→署名比較→必要時だけ裏で再ビルド
```

設計方針（内部仕様）
- 自動ビルド・自動再利用（破壊的変更でクリーン化）
  - 既存の `E.pipeline` を「宣言的ブループリント + 実行体」に統合し、ユーザーは `.build()` を書かない（呼んでも no-op/同義）。
  - 実行時: 入力スナップショット（t/cc/gui）を統合して params を解決→「解決後パラメータ署名」を生成→前回署名と一致なら既存 Pipeline を再利用→不一致なら裏で再ビルド。
  - Pipeline の単層 LRU はそのまま活用（キーは `(geometry_hash, pipeline_key)`）。`pipeline_key` は「解決後（量子化後）パラメータ」を材料に算出。

- 入力統合モデル（t/cc/gui は同一概念）
  - t/cc/gui を「ただの数値入力」として 1 つの InputContext に統合。ParameterRuntime は毎フレーム `set_inputs(t, cc_snapshot)` を受け取り、各 shape/effect の引数値を決める唯一の参照源（ParameterStore）を更新する（GUI は Store の override により自然適用）。
  - バインディングは単純化する。各パラメータは Descriptor ID で一意に識別され、値は ParameterStore に保持。GUI は `override`、MIDI は `midi_override` として同じパラメータへ上書き。`resolve()` は「original（ブループリント値）→midi_override→gui_override」の優先順で“現在値”を返す。
  - 時刻 t は「特別扱いしない数値パラメータ」。効果/形状が `t` を引数に持つ場合は、ParameterRuntime が自動で `t` を引数として供給する（関数に存在しない場合は渡さない）。

- パラメータ署名（ParamSignature）
  - 署名は「解決後（Runtime + 入力スナップショット適用後）の最終値」を `common.param_utils.params_to_tuple` で正規化。さらに浮動小数へ量子化（丸め）を適用。
  - 丸め粒度は `__param_meta__` の `step` またはヒューリスティク（既定 1e-3 など）で決定。ベクトルは各成分に適用。
  - `t` は“ただの入力”として扱い、結果的に有効パラメータが変わる場合のみ署名が変化→再ビルド。変わらなければヒット。

- 形状キャッシュ（GUI/Runtime 有効時も有効化）
  - これまで Runtime 介在中は LRU をバイパスしていたが、以後は「解決後の最終値」でキーを作成して LRU を使用する。
  - 具体: Runtime で `params_resolved = before_shape_call(..., inputs={t, cc, gui})` → `params_to_tuple(params_resolved)` を鍵に `_cached_shape` を使用。
  - 正しさ: CC/GUI/t を反映した最終値が LRU 鍵になるため、「ユーザーの見た目の値」とキャッシュの同一性が一致する。

- 丸め/ヒステリシス（裏側）
  - 丸め: ParamSignature 構築時に量子化（例: 1e-3 または `step`）。CC/GUI/t いずれの入力経由でも同じ扱い。
  - ヒステリシス: オプション。`±epsilon` 以内の揺れを同一と見なす（将来拡張）。

- 変更トリガ（自動無効化条件）
  - 署名の変化（最終値）/ ステップ配列の変更 / エフェクト関数バイトコードの変更 / 入力 Geometry.digest の変更。
  - これらは既存の `pipeline_key` と `geometry_hash` を流用しつつ、署名変化で裏ビルドを誘発。

- 並行性/安全性
  - `Pipeline` 内部で `RLock` を保持し、ビルドとヒット参照を最小限で保護。
  - 多スレッド/ワーカー共用は非推奨（従来どおり）。必要時はインスタンスを分ける。

API・後方互換（破壊的変更を許容してクリーン化）
- `E.pipeline` は「宣言 + 実行」を担う単一オブジェクトに統合（従来の Builder + Pipeline を統合）。
- `.build()` は省略可（互換のため残すが no-op/同義）。推奨は `pipe(g)` のみ。
- `pipeline_key` の定義変更: 「ビルド時の静的パラメータ」ではなく「解決後（量子化後）パラメータ」から算出。
- Runtime 介在時の Shapes LRU: バイパスを撤廃し、解決後キーで LRU を有効化。
- strict 機能は撤廃（API/実装ともに削除）。

設定・チューニング
- 環境変数/隠し設定（候補）
  - `PXD_PIPELINE_QUANT_STEP=1e-3`（丸め粒度の下限）。
  - `PXD_PIPELINE_HYSTERESIS=0.0`（将来拡張: ヒステリシス幅）。
  - `PXD_SHAPES_CACHE_WITH_RUNTIME=1`（Runtime 有効時の LRU を許可）。

受け入れ基準（DoD）
- GUI ON/OFF いずれでも、同じ `draw()` がそのまま動く。
- CC/GUI/t によって「有効パラメータが不変」の連続フレームでは、Effects/Shapes ともに再計算をスキップ（パイプラインはヒット、シェイプは LRU ヒット）。
- 有効パラメータが変わったフレームだけ再計算（裏ビルドが走り、その後は再度ヒット）。
- 既存のベンチにて `cache on/off` 差が維持され、回帰がない（`tests/perf/test_pipeline_perf.py`）。

- テスト計画（編集ファイル優先 / 追加予定）
- smoke: Pipeline の基本挙動（不変→ヒット / 変化→ミス→以後ヒット）。
- integration: GUI 有効時に Shape LRU が有効になること（解決後の値で鍵化）。
- perf: パイプライン・シェイプの miss/hit 比較ベンチ（既存パターンを流用）。

実装タスク（チェックリスト）
- [x] ParamSignature: `params_to_tuple` の上に量子化を追加（`__param_meta__.step` を優先）。
- [x] Effects: `E.pipeline` を「宣言 + 実行」統合オブジェクトへ刷新（署名比較／裏ビルド／RLock／`.build()` は同義）。
- [x] PipelineKey: 「解決後（量子化後）パラメータ」を材料に算出するよう変更。
- [x] Shapes: Runtime 介在時も「解決後パラメータ」で `_cached_shape` を使うよう経路変更。
- [x] Runtime: 入力スナップショット（t/cc）を `set_inputs()` で受け、ValueResolver が ParameterStore の値を一元解決（original→midi_override→gui_override）。
- [x] 設定: 丸め粒度（`PXD_PIPELINE_QUANT_STEP`）の環境変数を追加（任意）。
- [ ] ドキュメント: `architecture.md` のキャッシュ節を同期更新（鍵の再定義・`.build()` の扱い）。
- [ ] テスト: smoke/integration/perf の最小セット追加（変更ファイル優先で高速に回す）。
- [ ] デモ: `main2.py` に新 `pipeline` の最小例を追加（ユーザーの build 非認知例）。

## 実装計画（段階別チェックリスト・詳細）

段階0: 現状確認と影響範囲の固定
- [x] `src/api/effects.py` を精読し、`Pipeline`/`PipelineBuilder` の責務と依存（`_geometry_hash`, `_fn_version`, `_params_digest`, `RLock`, `OrderedDict`）を洗い出す。
- [x] `src/api/shapes.py` の Runtime 介在時 LRU バイパス箇所（200行前後）と `_cached_shape` 呼び出し経路を把握。
- [x] `engine/ui/parameters`（`runtime.py`/`value_resolver.py`/`state.py`）で、値解決・`__param_meta__` 取得・override の流れを把握。
- [x] `tools/gen_g_stubs.py` の E.pipeline/Builder/`.build()` の Stub 生成を確認（刷新時の同期対象）。
- [x] ベンチ/テストのキャッシュ参照箇所を確認（`tests/perf/test_pipeline_perf.py`）。

段階1: 量子化ユーティリティ（ParamSignature）
- [ ] `common/param_utils.py` に量子化ヘルパを追加（例: `quantize_params(params, meta, *, default_step=1e-3)`）。
  - 数値/ベクトルは `step`（`__param_meta__` にあれば優先、無ければ環境変数 `PXD_PIPELINE_QUANT_STEP` または既定値）で丸め。
  - 非数値はそのまま。最終的に `params_to_tuple` を用いた署名タプルを返す関数（`signature_tuple(params, meta)`）を用意。

段階2: Runtime の入力統合（t/cc/gui は同一の数値入力）
- [x] `ParameterRuntime` に `set_inputs(t, cc_snapshot)` を追加。
- [x] `ParameterManager.draw()` で `runtime.set_inputs(t, merged_cc)` を呼ぶ。
- [x] ValueResolver は ParameterStore の `original`（ブループリント）に対し、`midi_override`（CC）と `override`（GUI）を順に適用して“現在値”を返す。`t` は関数が `t` 引数を持つ場合にのみ追加で渡す。

段階3: Shapes — Runtime 下でも LRU を有効化
- [x] `ShapesAPI._build_shape_method()` を変更。
  - Runtime 有効時: `params_resolved = runtime.before_shape_call(...)` → 量子化して `params_tuple` → `return _cached_shape(name, params_tuple)`。
  - 二重解決回避: `_cached_shape()` は Runtime 非介在の実行経路（`_generate_shape_resolved(name, params_dict)`）を使用するよう変更。 [x]
- [ ] `_generate_shape()` を分割し、Runtime 無しで実行する経路を新設。ドキュメント/型注釈も更新。

段階4: Effects — `E.pipeline` の統合と裏ビルド
- [x] `PipelineBuilder` と `Pipeline` を統合し、`Pipeline`（ファサード）に一本化。
  - ステップ列（ブループリント）と直近の「解決後署名」、内部実行体 `self._compiled: _CompiledPipeline | None` を保持。
  - `.build()` は no-op（`return self`）。`.strict()` は撤廃。`.cache()` は存続。
-- [x] `_CompiledPipeline`（内部クラス）を新設。
  - 役割: 現在の「解決後（量子化後）パラメータ」で固定化された実行体。従来どおり LRU を持ち、初期化時に `pipeline_key` を計算。
  - `__call__(g)` は従来実装を踏襲（Runtime 介在不要）。
-- [x] `Pipeline.__call__(g)` を刷新。
  - Runtime で各ステップの params を解決 → 量子化 → 署名タプル生成。
  - 直近署名と比較し、異なれば `_CompiledPipeline` を再生成（裏ビルド）して差し替え。同一なら再利用。
  - `RLock` で差し替えと参照を保護。
- [x] `pipeline_key` 算出は「解決後（量子化後）パラメータ」を用いるよう変更（`_params_digest` の入力を署名由来に）。

段階5: API ・スタブ同期
- [x] `api/__init__.pyi` の `Pipeline/Builder` 型を統合仕様へ更新（`.build()` は同義）。
- [x] `tools/gen_g_stubs.py` の E.pipeline 生成部を更新。
- [x] `.strict()` のスタブ/実装を削除（互換レイヤは設けない）。
- [ ] 公開 API の例から `.build()` を省略する形に整理。

段階6: ドキュメント更新
- [ ] `architecture.md` のキャッシュ節を刷新（`pipeline_key` 再定義、`.build()` の位置づけ、Shapes LRU の変更）。
- [ ] 本提案 md の用語/手順を最終仕様に合わせて整える。

段階7: テスト
- [ ] smoke: Pipeline 連続呼び出し（CC/GUI/t 不変→ヒット、変化→裏ビルド→以後ヒット）。
- [ ] shapes: Runtime 介在時に `_cached_shape` が同一最終値でヒットすることを確認。
- [ ] perf: 既存ベンチ（`tests/perf/test_pipeline_perf.py`）を `.build()` 省略でも実行できるよう調整。
- [x] type/lint: 変更ファイルに限定し `ruff/black/isort/mypy` を通す。

段階8: デモ/サンプル
- [ ] `main.py`（または `main2.py`）を統合 `pipeline` に置換し、`.build()` を書かないサンプルへ更新。
- [ ] GUI ON/OFF の両方で挙動確認（必要最小のログで裏ビルドの発火が分かるように）。

段階9: 移行/クリーンアップ
- [ ] 旧 `PipelineBuilder` を削除/内部専用へ降格し、参照箇所/スタブ/テストを整理。
- [ ] `.strict()` の API/実装/テスト/ドキュメントを削除。
- [ ] 不要なヘルパや重複コメントを整理。docstring は日本語・NumPy スタイルに統一。
- [ ] 破壊的変更点を `CHANGELOG.md` に記録（無ければ新設）。

検証コマンド（変更ファイル優先）
- Lint: `ruff check --fix {changed}`
- Format: `black {changed} && isort {changed}`
- Type: `mypy {changed}`
- Test: `pytest -q -m smoke` または対象ファイルを直接指定

補足（トレードオフ）
- 量子化により「ごく小さな差」を同一視するため、超微細な連続操作では変化が吸収される。必要なら `__param_meta__.step` で粒度を下げる。
- Runtime 下での Shape LRU 有効化はメモリ保持が増える可能性があるため、`maxsize` などの制限を検討（別提案）。

確認事項（要合意）
- 公開名は従来どおり `E.pipeline`（Auto〜等の別名は導入しない）。
- 量子化の既定粒度: `1e-3` を初期値として妥当か（`__param_meta__.step` 優先）。
- GUI 有効時の Shape LRU キー: 「解決後の最終値」を正式キーにする方針で問題ないか。
- `main2.py` への導入例の掲載可否。

（この仕様とチェックリストでよければ、実装に着手します。追加/変更があればこの md を更新してから反映します。）
