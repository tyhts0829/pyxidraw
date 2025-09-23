# t/cc/gui 入力で shape/effect を駆動しつつキャッシュを効かせる改善計画（提案）

目的（ユーザー意図の再確認）

- draw 関数内で t/cc/gui を用いて shape と effect の各パラメータをコントロールできること（事前のビルド不要、宣言は薄い）。
- CC がプライマリ。GUI で値調整はできるが、その後のフレームで CC 値に更新が入れば値飛びしてよい（CC の物理つまみ位置を真とする）。
- GUI は常に有効化可能で、CC の状態が GUI 側にも反映される。
- t/cc/gui による「有効パラメータに変化がない」フレームでは、shape/effect ともにキャッシュを効かせて再計算を避ける。

方針の要点

- 入力統合: t/cc/gui を単一モデルの“数値入力”として扱う（既存の Runtime.set_inputs(t, cc) を継続使用、GUI は ParameterStore の override から自然適用）。
- バインディング（重要）: どの入力をどのパラメータに適用するかは「draw 関数内でユーザーが決める」。`cc` は `{index(int): value(0..1)}` の辞書で、ユーザーが `effect(param=cc[1])` のように割り当てる。
- 優先度: CC（midi_override） > GUI（override） > original（ブループリント値）。
- 署名とキャッシュ: 解決後（t/cc/gui 適用＋量子化後）の値で署名を作成し、Shapes LRU / Pipeline 裏ビルド & LRU に反映（不変フレームはヒット）。

推奨する割当方法（シンプルかつ明示的）

- 直接値渡し（最小）: `amplitude = cc.get(1, 0.0)` のように数値を直接渡す。
  - GUI 調整中に CC が更新されると GUI の override に勝てない場合がある（優先度実装を CC 優先に変えるだけでは“どの値が CC 由来か”を識別できないため）。
- CC マーカー（推奨）: `amplitude = CC(1)` のように「値ソース=CC」を明示するラッパを用意し、Resolver が `midi_override` として扱う。
  - ベクトルも `scale=(CC(2), 1.0, 1.0)` のようにコンポーネント単位で指定可能。
  - 変換（0..1→ 実レンジ）はユーザー式 or `CC(1, map=lambda v: v*2.0)` で対応（将来 `scale/offset` 等の糖衣も可）。

仕様（ドラフト）

- バインディングの責務分離
  - ユーザーは draw 内で `cc`/`t` を任意に組み合わせて shape/effect 引数へ渡す（例: `E.pipeline.affine(scale=(1+cc[2], 1, 1))`）。
  - エンジン側は「渡された最終値」をそのまま採用し、GUI の override/反映は ParameterStore の既存仕組みで行う。
- 優先度（重要）
  - ParameterStore.resolve() の優先順を `midi_override > override > original` に変更（現状は override > midi_override）。
  - これにより、GUI 調整後も CC 更新が入れば値は CC にスナップ（“値飛び可”の要件を満たす）。
- 量子化
  - `common.param_utils.signature_tuple(params, meta)` を使用（既存）。
  - `meta.step` を最優先、無ければ `PXD_PIPELINE_QUANT_STEP`、無指定は 1e-3。
  - 量子化は署名生成時に行い、CompiledPipeline 固定値にも適用（一貫性）。
- キャッシュ
  - Shapes: Runtime 下でも「解決後（量子化後）」のキーで LRU（既存実装）。
  - Effects: 実行時に「解決後（量子化後）」の署名を比較 → 差異があれば裏ビルドして pipeline_key を更新（既存実装）。

非目標（今回はやらない）

- 高度なカーブ/モード（ホールド/スムージング/デッドゾーン）。必要があれば別提案。
- GUI 側の双方向バインディング設定 UI（マッピング UI は非目標。バインディングは draw 内で記述）。

設計詳細

- draw 内バインディング
  - ユーザーが `cc`/`t` を直接引数に割り当てる。エンジンはこの“解決済みの最終値”を受け取り、量子化 → 署名 → キャッシュへ反映するだけ（cc→param の自動バインディングは行わない）。
- t の取り扱い
  - シグネチャに `t` が存在する場合のみ注入（現状どおり）。
  - 署名には「t 由来で実際にパラメータへ寄与した値」のみ反映（t が引数でない場合は署名に影響しない）。

インターフェース要約（変更点）

- ParameterStore.resolve(): 優先度を `midi_override > override > original` に変更（破壊的）。
- draw 内バインディングを前提に、`CC(idx, map=...)` マーカーを用意。Resolver は CC マーカーを検出して `midi_override` を設定、数値はその場で解決。
- 既存の Runtime.set_inputs(t, cc) を継続使用（GUI は既存の override チャネルで反映）。

影響とリスク

- 優先順変更により、既存の「GUI が優先」挙動から「CC が優先」に変わる（破壊的）。
- CC 継続更新環境では GUI での微調整が上書きされやすい（意図どおりだが周知要）。
- バインディングがないパラメータは従来どおり（original を使用）。

オプション（将来）

- cc_curve（指数/対数/カスタム LUT）
- デッドゾーン/ヒステリシス/ホールド（短時間の GUI 操作を尊重、等）
- GUI からのバインディング指定 UI（Editor）

確認事項

- CC マーカーの API（`CC(idx, map=...)`）の表記と糖衣（scale/offset）をどこまで用意するか -> CC はただの辞書だから API とか要らんよ。
- 初期の step 既定（1e-3 で問題ないか）→OK
- ドキュメント更新の範囲（README/architecture.md/サンプル）→ すべて。

---

実装計画（詳細）

1. CC 優先の解決順（破壊的変更）

- [x] src/engine/ui/parameters/state.py: ParameterStore.resolve()
  - 既存: override → midi_override → original
  - 変更: midi_override → override → original
  - 影響: GUI 調整後も CC 更新で値ジャンプ（仕様どおり）。
  - テスト: tests/ui/parameters/test_parameter_store.py に優先順の検証を追加。

2. CC マーカーの導入（draw 内バインディングを明確化）

- [x] 新規: src/engine/ui/parameters/cc_binding.py
  - `class CCBinding`: (index: int, map: Callable[[float], float] | None)
  - `def CC(index: int, map: Callable[[float], float] | None = None) -> CCBinding`
  - ユーザーは draw 内で `CC(1)`/`CC(2, map=...)` を引数へ埋め込む。
- [x] ValueResolver.resolve()（src/engine/ui/parameters/value_resolver.py）
  - スカラー: 値が CCBinding の場合、`runtime` の cc スナップショットから値を取得 →map 適用 →store.register(... original は据え置き)→store に `midi_override` をセット（要 API）→ 返却値に適用。
  - ベクトル: 各コンポーネントが CCBinding なら同様に処理、数値/CC 混在を許容。
  - 数値/文字列などは従来通り。
- [x] ParameterStore に midi_override をセットする API を追加（必要なら）
  - 既存の `set_override(param_id, value, source="midi")` を使用するため追加不要。
- [x] Runtime.set_inputs(t, cc) は現状のまま使用（cc は 0..1 正規化）。

3. 量子化と署名（既存の適用範囲を明確化）

- [x] 量子化: `common.param_utils.signature_tuple()` を Effects/Shapes 双方で使用済み（検証のみ）。
- [x] step 未指定は env `PXD_PIPELINE_QUANT_STEP`→1e-3。
- [x] CompiledPipeline は量子化後の値で固定化（現状維持）。

4. ドキュメント/サンプル

- [ ] docs/proposals/user_transparent_effect_cache.md: 「draw 内バインディング（CC マーカーの使用）」を明記。
- [ ] main2.py: CC → effect/shape 割当の具体例（スカラー/ベクトル + map 使用例）。

5. 検証（変更ファイル優先）

- [x] type/lint: mypy/ruff を変更ファイルに限定して通す。
- [ ] smoke: CC マーカー使用時に GUI override より CC が優先されることの確認（headless で store.resolve を直接検証）。
- [ ] perf: CC 不変フレームでヒット、変化時にミス → 以後ヒットの確認。
