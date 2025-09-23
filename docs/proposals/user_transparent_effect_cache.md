# ユーザー非認知のキャッシュ運用（Effects/Shapes）仕様（提案）

目的
- ユーザーにキャッシュや build の存在を意識させず、CC/GUI を含む通常操作で「変わらないフレームは再計算しない」を実現する。
- GUI 有効時でも Shape 生成にキャッシュを効かせ、全体のレイテンシとスループットを改善する。

スコープ（対象と非対象）
- 対象: `api.effects` の Pipeline 実行、`api.shapes` の生成、`engine.ui.parameters`（Runtime 経由のパラメータ適用）。
- 非対象: 外部ファイル入出力・レンダラ・MIDI 実装詳細（CC 値供給は既存の snapshot に依存）。

達成したい UX（ユーザー視点の要件）
- ユーザーは build やキャッシュ戦略を呼び分けない。常に同じ書き方でよい。
- CC/GUI でパラメータを変更したフレームだけ再計算。変更がなければ全ステップをスキップ（即時ヒット）。
- GUI を ON にしても Shape 側の LRU が効く（＝生成済み形状の再利用で安定レイテンシ）。
- 丸め（量子化）やヒステリシス等の安定化は裏側で自動適用。ユーザーはしきい値を意識しない。

動作イメージ（main2.py を想定）
```python
from api import G, E

# 例: CC に応じて affine を操作、ユーザーは build を書かない
auto = E.autopipeline(lambda P: P.affine().extrude(distance=10, direction=(0,0,1)))

def draw(t, cc):
    g = G.polyhedron().scale(200).translate(200, 200, 0)
    return auto(g)  # ← 内部で Runtime 解決→署名比較→必要時だけ再ビルド
```

設計方針（内部仕様）
- 自動ビルド・自動再利用
  - `AutoPipeline`（仮称）を導入。ビルダー相当のステップ定義（ブループリント）と、直近の「解決後パラメータ署名」を保持。
  - 実行時: Runtime 経由で params を解決→署名生成→前回署名と一致なら既存 Pipeline を再利用→不一致なら裏で `build()` して差し替え。
  - Pipeline の単層 LRU はそのまま活用（キーは従来の `(geometry_hash, pipeline_key)`）。

- パラメータ署名（ParamSignature）
  - 署名は「解決後（Runtime 適用後）の最終値」を `common.param_utils.params_to_tuple` で正規化。さらに浮動小数へ量子化（丸め）を適用。
  - 丸め粒度は `__param_meta__` の `step` またはヒューリスティク（既定 1e-3 など）で決定。ベクトルは各成分に適用。
  - 署名が変わらない限り Pipeline は再ビルドしない（＝CC 不変フレームは即ヒット）。

- 形状キャッシュ（GUI/Runtime 有効時も有効化）
  - これまで Runtime 介在中は LRU をバイパスしていたが、解決後の最終値でキーを作成して LRU を使用する。
  - 具体: Runtime で `params_resolved = before_shape_call(...)` 実行→`params_to_tuple(params_resolved)` を鍵として `_cached_shape` を経由。
  - 正しさ: Runtime の override/CC/GUI を反映した最終値が LRU 鍵になるため、「ユーザーの見た目の値」とキャッシュの同一性が一致する。

- 丸め/ヒステリシス（裏側）
  - 丸め: ParamSignature の構築時に量子化（例: 1e-3 または `step`）。
  - ヒステリシス: オプション。`±epsilon` 以内の揺れを同一と見なす（将来拡張）。

- 変更トリガ（自動無効化条件）
  - 署名の変化（最終値）/ ステップ配列の変更 / エフェクト関数バイトコードの変更 / 入力 Geometry.digest の変更。
  - これらは既存の `pipeline_key` と `geometry_hash` を流用しつつ、署名変化で裏ビルドを誘発。

- 並行性/安全性
  - `AutoPipeline` 内部で `RLock` を保持し、ビルドとヒット参照を最小限で保護。
  - 多スレッド/ワーカー共用は非推奨（従来どおり）。必要時はインスタンスを分ける。

API・後方互換
- 追加（候補）
  - `api.effects.AutoPipeline` と `E.autopipeline`: ユーザーはブループリント（`lambda P: P.affine()....`）を渡すだけ。
  - 既存の `E.pipeline` はそのまま（明示ビルドの低レベル API として維持）。
- 既存コードの互換性
  - 変更不要。必要なら `AutoPipeline` を任意に採用できる。

設定・チューニング
- 環境変数/隠し設定（候補）
  - `PXD_AUTOBUILD_QUANT_STEP=1e-3`（丸め粒度の下限）。
  - `PXD_AUTOBUILD_HYSTERESIS=0.0`（将来拡張: ヒステリシス幅）。
  - `PXD_SHAPES_CACHE_WITH_RUNTIME=1`（Runtime 有効時の LRU を許可）。

受け入れ基準（DoD）
- GUI ON/OFF いずれでも、同じ `draw()` がそのまま動く。
- CC/GUI で値を変えない連続フレームでは、Effects/Shapes ともに再計算をスキップ（パイプラインはヒット、シェイプは LRU ヒット）。
- 値を変えたフレームだけ再計算（Pipeline 裏ビルドが走り、その後は再度ヒット）。
- 既存のベンチにて `cache on/off` 差が維持され、回帰がない（`tests/perf/test_pipeline_perf.py`）。

テスト計画（編集ファイル優先 / 追加予定）
- smoke: AutoPipeline の基本挙動（不変→ヒット / 変化→ミス→以後ヒット）。
- integration: GUI 有効時に Shape LRU が有効になること（解決後の値で鍵化）。
- perf: パイプライン・シェイプの miss/hit 比較ベンチ（既存パターンを流用）。

実装タスク（チェックリスト）
- [ ] ParamSignature: `params_to_tuple` の上に量子化を追加（`__param_meta__.step` を優先）。
- [ ] Effects: `AutoPipeline` 実装（ブループリント保持／署名比較／裏ビルド／RLock）。
- [ ] Shapes: Runtime 介在時も「解決後パラメータ」で `_cached_shape` を使うよう経路変更。
- [ ] 設定: 丸め粒度・ヒステリシス・Runtime 時 LRU 許可の環境変数追加（任意）。
- [ ] ドキュメント: `architecture.md` のキャッシュ節を同期更新（新モード説明と鍵の定義を追記）。
- [ ] テスト: smoke/integration/perf の最小セット追加（変更ファイル優先で高速に回す）。
- [ ] デモ: `main2.py` に AutoPipeline の最小例を追加（ユーザーの build 非認知例）。

補足（トレードオフ）
- 量子化により「ごく小さな差」を同一視するため、超微細な連続操作では変化が吸収される。必要なら `__param_meta__.step` で粒度を下げる。
- Runtime 下での Shape LRU 有効化はメモリ保持が増える可能性があるため、`maxsize` などの制限を検討（別提案）。

確認事項（要合意）
- AutoPipeline の公開名: `E.autopipeline` でよいか（別名: `E.auto`）。
- 量子化の既定粒度: `1e-3` を初期値として妥当か（`__param_meta__.step` 優先）。
- GUI 有効時の Shape LRU キー: 「解決後の最終値」を正式キーにする方針で問題ないか。
- `main2.py` への導入例の掲載可否。

（この仕様とチェックリストでよければ、実装に着手します。追加/変更があればこの md を更新してから反映します。）

