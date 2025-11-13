どこで: src/ 配下全体（設計・対称性・一貫性レビュー）
何を: アーキテクチャ、shape/effect の対称性、実装の一貫性を厳しめに点検
なぜ: 現状の設計品質を客観評価し、将来の改善点を明確化するため

---

# 総評（TL;DR）

- 層分離（L0–L3）と依存方向は概ね良好。engine/* は registries に触れず、API 層が関数参照を注入する設計が徹底されている（lazy 署名・キャッシュも整合）。
- shape/effect は関数ベース＋レジストリの対称設計が実現されており、G/E の利用体験は一貫している（遅延評価・署名・量子化の挙動も仕様通り）。
- 実装の一貫性も概ね良いが、少数の重複・未使用引数・初期化重複・docstring/RangeHintのばらつきがある。優先度中で解消を推奨。

---

# アーキテクチャ（分離・依存・キャッシュ）

- 層分離と依存方向
  - architecture.md のキャンバスと実装の整合は概ねクリア。engine/* から effects/*, shapes/* へ直接依存は見当たらず、API 層（G/E）が注入する関数参照のみで lazy を駆動している。
  - 参照: architecture.md:12, src/engine/render/AGENTS.md:9, src/engine/core/lazy_geometry.py:1

- LazyGeometry の評価とキャッシュ
  - base が shape-spec のとき、実体化時に params_signature による鍵を用いて LRU 共有（src/engine/core/lazy_geometry.py:25, 41, 59）。
  - plan（effects）は impl_id + 量子化パラメータで prefix LRU を構成（途中結果再利用）。ObjectRef 除外も仕様通り（src/engine/core/lazy_geometry.py:82, 101）。
  - 署名の外部公開は api.lazy_signature.lazy_signature_for に統一（src/api/lazy_signature.py:29）。

- ランタイム/パラメータ解決
  - ParameterRuntime が before_shape_call/before_effect_call で実値/量子化・bypass を解決（src/engine/ui/parameters/runtime.py:56, 97, 130）。
  - style は非幾何エフェクトとして worker 側で抽出し描画レイヤへ変換（src/engine/runtime/worker.py:1, 66）。architecture.md の説明と一致。

結論: コア設計（spec→lazy→realize）とキャッシュ戦略は architecture.md と整合し、境界の摩擦が少ない。小さな初期化重複等を除けば健全。

---

# shape/effect の対称性（API/登録/量子化/バイパス）

- レジストリと API 対称性
  - BaseRegistry によるキー正規化と登録/取得 API を共有（src/common/base_registry.py:1）。
  - shapes.registry と effects.registry は対称 API（@shape/@effect, get/list/is_registered/clear/unregister）で統一（src/shapes/registry.py:1, src/effects/registry.py:1）。
  - 公開入口も対称: `G`（ShapesAPI）と `E.pipeline`（PipelineBuilder）（src/api/shapes.py:1, src/api/effects.py:1）。

- 遅延評価の対称性
  - G.<name>(...) は LazyGeometry(base_kind="shape") を返す（src/api/shapes.py:96, 166）。
  - E.pipeline.<fx>(...) は plan に (impl, params) を積み、build() で Pipeline。Pipeline(g) は Lazy を連結（src/api/effects.py:120, 148, 172）。

- 量子化の非対称（仕様通り）
  - Shapes: 署名のみ量子化、実行は非量子化（runtime 解決値）を渡す（src/api/shapes.py:176, src/engine/core/lazy_geometry.py:48, 69）。
  - Effects: 量子化後の値が実行引数にも渡る（PipelineBuilder が params_signature を dict 化して保存）（src/api/effects.py:184, 190）。

- 共通バイパス（Effects のみ）
  - effects.registry が impl に __effect_impl__/__effect_supports_bypass__ を付与（src/effects/registry.py:33）。
  - ランタイムと PipelineBuilder が bypass を統合し、true ならステップ自体を追加しない（src/engine/ui/parameters/runtime.py:158, src/api/effects.py:136）。

結論: 期待通りの対称設計＋必要な非対称（量子化/バイパス）が明確に実装されている。

---

# 実装の一貫性チェック（抜き取り）

- Geometry の不変条件/型
  - coords=float32(N,3), offsets=int32(M+1) を強制。as_arrays(copy=False) は読み取り専用ビューを返す（src/engine/core/geometry.py:69, 167）。
  - 変換は純関数で新インスタンスを返す（translate/scale/rotate/concat）（src/engine/core/geometry.py:183, 219, 259, 307）。

- エフェクトの出力 dtype/処理
  - 代表例 dash: 計算は float64、出力は float32 に統一（src/effects/dash.py:1, 206）。no-op 経路も明示。
  - translate は Geometry のメソッドへ委譲し、空/ゼロの高速経路あり（src/effects/translate.py:1）。

- RangeHint/__param_meta__
  - 多くの shape/effect に定義あり。一方で step 未指定が散見（既定 1e-6 にフォールバック）（例: src/shapes/lissajous.py:24）。

- スタイル（非幾何エフェクト）
  - 関数は no-op、__effect_kind__=style マーカーで worker が抽出（src/effects/style.py:1, 36）。

- 一貫性に関する気づき（要是正）
  - api/effects 内のグローバル変数初期化が重複している（再代入が2箇所）: src/api/effects.py:242, 265。単一点に集約を推奨。
  - core の内蔵軽量エフェクトで未使用引数: _fx_scale/_fx_rotate の auto_center が未使用（src/engine/core/lazy_geometry.py:286, 303）。署名から外すか、意味を持たせるべき。
  - docs/docstring のばらつき: 公開 API に NumPy スタイルを徹底しきれていない箇所がある（例: src/shapes/grid.py:29 は簡潔説明のみ）。

---

# リスク/懸念（優先度付き）

1) 軽微: グローバル初期化の重複（api/effects）
   - 影響: 機能的には問題になりにくいが、可読性・意図の明確さを損なう。
   - 対応: WeakSet/OrderedDict の初期化をモジュール末尾の 1 箇所に統合。

2) 軽微: 内蔵 fx の未使用引数（auto_center）
   - 影響: 仕様読みの混乱を生む。将来の意図がないなら削除が望ましい。

3) 軽微〜中: RangeHint の step 未指定のばらつき
   - 影響: 署名量子化の安定性（キャッシュ効率）に揺らぎ。UI スライダの体感ステップも不統一。
   - 対応: よく使う数値パラメータへ step を付与（vec の場合は成分ごと）。

4) 軽微: 不要ユーティリティの残存
   - 例: api/effects._is_json_like は未使用（src/api/effects.py:272）。削除/テスト補助に用途限定する。

---

# 改善提案（最小・非破壊での質向上）

- 1. api/effects のグローバル初期化を一箇所へ集約
  - 目的: 可読性と初期化順序の明確化。
  - 対応: _GLOBAL_PIPELINES/_GLOBAL_COMPILED の try/except 初期化を末尾の一箇所に統合し、前方参照は Optional を前提にガード。

- 2. core の内蔵 fx から未使用引数を除去 or TODO を明記
  - 目的: 署名と実装の乖離を無くす。
  - 対応: auto_center を削るか、中心計算を実装するかのどちらかに統一（後者なら Geometry 側に center-of-mass 補助を追加）。

- 3. RangeHint の step を重点パラメータへ追加
  - 目的: キャッシュ鍵の安定化と UI 体験向上。
  - 対応例: lissajous.freq_*, grid.nx/ny, wobble.amplitude/frequency/phase 等に実用的な step を設定。

- 4. 公開 API の docstring を NumPy スタイルへ統一
  - 目的: 利用者ドキュメントの明確化。自動ドキュメント生成の基盤。
  - 対応: 変更規模は小（関数コメント中心）。

---

# architecture.md との差分候補（更新提案）

- Effects の「量子化後の値が実行引数にも渡る」旨は明記済みだが、G と E の差異（Shapes は実行は非量子化）をもう一段強調しても良い（誤解防止）。
- style の扱い（worker で抽出・plan から除去）は十分に書かれているが、__effect_kind__=style という検出契約を1文追記すると発見性が上がる（src/effects/style.py:36, src/engine/runtime/worker.py:15）。

---

# 参考コード箇所（抜粋）

- レジストリ基底: src/common/base_registry.py:1
- shapes レジストリ: src/shapes/registry.py:1
- effects レジストリ: src/effects/registry.py:1
- G（ShapesAPI）: src/api/shapes.py:1, src/api/shapes.py:166
- E（PipelineBuilder）: src/api/effects.py:120, src/api/effects.py:184
- Lazy 実体化/キャッシュ: src/engine/core/lazy_geometry.py:25, src/engine/core/lazy_geometry.py:82
- Geometry 不変条件: src/engine/core/geometry.py:30, src/engine/core/geometry.py:167
- style の抽出: src/effects/style.py:36, src/engine/runtime/worker.py:1

---

# 改善アクション案（要確認・チェックリスト）

- [ ] api/effects: グローバル初期化の重複を削除して一箇所に集約
- [ ] engine/core/lazy_geometry: _fx_scale/_fx_rotate の auto_center を整理（削除 or 実装）
- [ ] RangeHint: 主要パラメータへ step を付与（float/vector）
- [ ] Docstring: 公開 API（特に shapes/）を NumPy スタイルに統一
- [ ] api/effects: 未使用の _is_json_like を削除（必要なら tests/helper へ移動）

上記の具体修正に進める場合は、このチェックリストを基に作業します。優先順位や step 値の粒度など、先にご希望があれば指示してください。

