# architecture.md 同期監査レポート（2025-11-07）

本レポートは、ルート `architecture.md` の記述と実装（`src/` 配下）の整合性を徹底確認した結果のまとめ。重要な一致点と乖離点、更新提案を示す。

## 概要

- 対象範囲: ルート `architecture.md` と `src/` 以下の全モジュール
- 手法: 依存関係の逆探索（rg）、仕様項目ごとのコード参照、実装内ドキュメントの照合
- まとめ: 大部分は整合。特にレイヤリング、投影行列、Lazy 化・署名ベース設計、GUI/ワーカー分離は一致。主な乖離は「Geometry.digest の廃止」「パイプライン中間キャッシュの所在」「ModernGL の optional 記述」と、少数の規約逸脱（print 利用）

## 一致点（抜粋）

- 層構造と依存方向（外→内、循環なし）は概ね順守
  - engine→api/effects/shapes 参照なし（ドキュメント内コード例を除く）
  - shapes/effects は `engine.core.geometry` のみ参照（L0）
- 投影・座標系（mm 基準、正射影、Y 反転）は実装どおり
  - `api/sketch_runner/utils.py:62` の `build_projection` が仕様の 4x4 行列を構築
  - `api/sketch_runner/render.py:55` でブレンド設定（SRC_ALPHA, ONE_MINUS_SRC_ALPHA）も反映
- Geometry の統一表現・不変条件・読み取り専用ビュー
  - 正規化・不変条件検証と `as_arrays(copy=False)` の読み取り専用ビュー化は実装済み（`src/engine/core/geometry.py:81`/`186`）
- Lazy 化・署名ベースのキャッシュ設計
  - 形状結果 LRU とエフェクト prefix LRU は `LazyGeometry` 側に実装（`src/engine/core/lazy_geometry.py:49`/`78` ほか）
  - Lazy 署名は `api.lazy_signature` による決定的 128bit（`src/api/lazy_signature.py:33`）
- GUI/ランタイム分離と Dear PyGui 駆動
  - `pyglet.clock.schedule_interval` 優先、未導入時はバックグラウンドスレッド（`src/engine/ui/parameters/dpg_window.py:1315`）
- 並行処理（WorkerPool/StreamReceiver/SwapBuffer）の責務と流れは記載通り
  - `src/engine/runtime/worker.py` / `receiver.py` / `buffer.py` を確認
- Indices LRU（offsets 署名ベース）と 1 ドロー設計
  - `renderer._geometry_to_vertices_indices()` と LRU 実装（`src/engine/render/renderer.py:247` 以降）

## 重要な乖離点（要ドキュメント更新）

1) Geometry.digest の扱い（記述あり / 実装なし）
- 記述（architecture.md）では `Geometry.digest: bytes` を前提にキャッシュ鍵へ利用、`PXD_DISABLE_GEOMETRY_DIGEST` も言及。
- 実装では digest を廃止し、キャッシュは Lazy 署名へ完全移行。
  - `src/engine/core/geometry.py:116` に「digest は廃止」と明記。
  - パイプラインキャッシュは `lazy_signature_for(lg)` を鍵に使用（`src/api/effects.py:90`）。
  - 形状/中間結果は `LazyGeometry` 側の LRU で管理（`src/engine/core/lazy_geometry.py:49`/`78`）。
- 影響: architecture.md の digest 関連節（ダイジェスト仕様/無効化環境変数/geometry_hash の説明）は「Lazy 署名」へ置換が必要。

2) パイプラインの中間キャッシュの所在
- 記述は「CompiledPipeline ローカル LRU（prefix 単位）」と Pipeline 側にあるように読める箇所あり。
- 実装では prefix LRU は `LazyGeometry.realize()` 内（engine.core）で適用。Pipeline は「署名→結果」の単層 LRU のみ。
  - `src/api/effects.py:90-105`（Pipeline の LRU）、`src/engine/core/lazy_geometry.py:83-137`（prefix LRU）
- 影響: 中間キャッシュは Lazy 側にある旨へ表現を調整。

3) ModernGL の optional 扱いの表現
- 記述では optional 方針の一般論内に ModernGL も含めた説明があり、局所 import で劣化運用を示唆。
- 実装は ModernGL を実行時前提としてトップレベル import している箇所がある。
  - `src/engine/render/renderer.py:14`、`src/api/sketch_runner/render.py:11`
- ただし AGENTS.md（ルート）では「実行時依存: ModernGL は必須」と明示。
- 影響: architecture.md の ModernGL に関する optional 風の記述は、AGENTS.md に合わせて「必須（init_only=True で GL 初期化回避のみ可）」へ改訂推奨。

4) print() 禁止規約の違反
- 記述は `print()` 禁止・logging 経由を要求。
- 実装にデバッグ用の print が残存。
  - `src/util/fonts.py:101`（環境変数ガード付き）。
- 影響: logging 置換または削除の必要（軽微）。

5) Export 画像仕様の粒度
- 記述は「HUD 含む低コスト（画面バッファ）/HUD なし高品質（FBO）」の二経路を説明。
- 実装は両経路を実装済み（`include_overlay=True/False`）で整合。モジュール先頭のコメントには旧段階の注記が残るが、architecture.md とは矛盾せず、ドキュメント側は現実装と一致。
  - `src/engine/export/image.py:54-99`（画面バッファ保存）、`100-153`（FBO 描画保存）
- 影響: 実装モジュールの先頭コメントを現状に合わせて更新すると更に明快。

## 仕様の明確化/調整が望ましい点（軽微）

- 「Optional Dependencies（方針）」章の ModernGL 位置づけを AGENTS.md と統一（必須へ）。
- 「幾何ダイジェスト」節を「Lazy 署名（base + plan）」に置換し、鍵の構成（impl_id + 量子化パラメータ署名）を明記。
- 「パイプライン厳格検証」節は現状（build 時は行わず、実行時検出）に合致しており、補足として ParameterRuntime による `__param_meta__` 利用と量子化の経路を追記すると親切。

## 依存境界・設計ルールの監査

- engine/* → api/* 参照: 実行コードでは該当なし（docstring の例外のみ）。
  - 検索結果: 該当ヒットは docstring（例）のみ（`src/engine/core/geometry.py:63`）。
- engine/* → effects/*|shapes/* 参照: なし。
  - 検索結果: 該当なし（レンダ/UI/IO/Runtime 層からの直接参照は未検出）。
- effects/*, shapes/* → engine/render|runtime|ui|io 参照: なし。engine.core.geometry のみ参照。
  - 実例: `src/effects/rotate.py:18` など、多数が `engine.core.geometry` のみ参照。
- common/*, util/* が上位層へ依存: なし。
- 以上、層ルールは合格。

## 参考（主な仕様→実装の照合箇所）

- 投影行列とユニフォーム
  - `src/api/sketch_runner/utils.py:62`（行列定義）、`src/engine/render/renderer.py:55`（uniform 書き込み）
- Indices LRU と PR index
  - `src/engine/render/renderer.py:247` 以降（LRU 設定/統計）、`src/util/constants.py:30`（PRIMITIVE_RESTART_INDEX）
- LazyGeometry の prefix LRU / 形状 LRU
  - `src/engine/core/lazy_geometry.py:49`（shape LRU）、`78`（prefix LRU）
- パラメータ量子化・署名
  - `src/common/param_utils.py:120` 以降（量子化/署名タプル）、`src/api/lazy_signature.py:33`（署名集約）
- GUI 駆動
  - `src/engine/ui/parameters/dpg_window.py:1315`（pyglet 駆動 or thread）

## 更新提案（architecture.md 差分案の要点）

- 「Geometry.digest」節を「Lazy 署名」へ全面置換（鍵の構成と無効化可否の説明を更新）。
- 「パイプラインの中間キャッシュ」節は Lazy 側に所在する旨へ修正。
- 「ModernGL」扱いを「必須（init_only=True で GL 初期化スキップ可）」へ明記し、トップレベル import 設計に合わせる。
- 付随で `print()` 禁止の遵守確認と `src/util/fonts.py:101` の置換/削除をタスク化。

## 補足（今後の小改善候補）

- `src/engine/export/image.py` のモジュール先頭コメントを現状に合わせて更新（FBO 経路あり）。
- 監視/HUD/GUI のフォント探索に関するドキュメントを docs 配下に分離し、設定項目との対応表を簡潔に提示。

---

本レポートは現状の実装に対しての監査結果であり、architecture.md を更新する際は上記「更新提案」をベースに、当該ファイル内の該当節へコード参照付きで反映すると整合が取れる。

