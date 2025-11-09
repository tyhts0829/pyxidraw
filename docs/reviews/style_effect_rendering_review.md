どこで: docs/reviews（仕様・実装レビュー）
何を: エフェクト `style` が実際の描画（色/太さ）に反映されるまでの処理フローをコードベースで追跡し、設計/実装の健全性と注意点をレビュー。
なぜ: 非幾何エフェクト（色/太さ）の適用地点を明確化し、GUI/ランタイム/レンダラ間の責務分担と落とし穴を把握するため。

**概要（結論）**
- `style` は幾何に手を触れない no-op（後勝ち）として宣言され、ワーカで `LazyGeometry.plan` から抽出→`StyledLayer` へ変換、メインでレイヤー逐次描画時に `set_line_color`/`set_line_thickness` を適用する設計。
- GUI の `style.*` は「draw 内で未指定の引数のみ」上書き対象。実際の値適用はワーカ実行直前にスナップショットをランタイムへ注入する（パイプラインは毎フレーム宣言が推奨）。
- 幾何キャッシュに影響を与えないため、`style` 変更のみでは形状再計算を誘発しない（計算効率の観点で良好）。

**フロー詳細（コード参照付き）**
1) パイプライン宣言とランタイム解決（GUI/CC 反映）
   - 利用側は `E.pipeline.style(color=..., thickness=...)` をチェーン宣言。
   - `PipelineBuilder.__getattr__` がアドホックな `effect` 呼び出しを受け取り、アクティブな `ParameterRuntime` があれば `before_effect_call(...)` で値を解決・登録（未指定のみ GUI 反映）。
     - 参照: `src/api/effects.py:127-153`, `src/engine/ui/parameters/runtime.py:104-127`
   - スナップショット適用はワーカ側で行う（後述）。このため「GUIでの動的変更を反映させたい場合、パイプラインは draw() 内で毎フレーム宣言する」のが前提（設計ガイドに整合）。

2) `style` の性質（非幾何・後勝ち）
   - 実装は no-op で `Geometry` をそのまま返す。メタ情報のみ付与。
     - 参照: `src/effects/style.py:19`（関数定義）、`src/effects/style.py:37`（`__param_meta__`）、`src/effects/style.py:50`（`__effect_kind__ = "style"`）
   - パラメータは `color: vec3 (0..1)` と `thickness: float (1..10)`。キャッシュ署名は `step` に従い量子化（色は 1/255 単位）。
     - 参照: `src/common/param_utils.py:101-159`（量子化/署名）

3) ワーカでの正規化（レイヤー化）
   - `draw(t)` の戻り値（`Geometry`/`LazyGeometry`/その配列）を `_normalize_to_layers(...)` でレイヤー列へ正規化。
     - 参照: `src/engine/runtime/worker.py:75-83`
   - `LazyGeometry` の `plan` を走査し、`__effect_kind__ == "style"`（または `__name__ == "style"`）を検出して「最後の指定」を採用。`style` ステップは plan から除去して幾何専用の plan を再構築。
     - 参照: 検出 `src/engine/runtime/worker.py:64-72`、抽出/除去/適用 `src/engine/runtime/worker.py:96-120`, `125-151`
   - 抽出した `color` は `util.color.normalize_color` で RGBA(0..1) へ正規化。`thickness` はグローバル基準太さに乗算する倍率として保持。
     - 参照: `src/util/color.py:35-73`
   - 出力は `RenderPacket(layers=tuple[StyledLayer,...])` としてメインへ送出（geometry は None 推奨）。
     - 参照: `src/engine/render/types.py:19-24`, `src/engine/runtime/packet.py:16`

4) メインスレッドへの受け渡し
   - `StreamReceiver` が結果キューから最新フレームのみ取り出し、`layers` があればそのまま `SwapBuffer.push(list(layers))`。
     - 参照: `src/engine/runtime/receiver.py:53-68`, `src/engine/runtime/buffer.py:33-41`

5) レンダラでの逐次描画
   - `LineRenderer.tick()` は `SwapBuffer` から `list[StyledLayer]` を検知すると `_frame_layers` に保持してアップロードを defer。
     - 参照: `src/engine/render/renderer.py:112-127`
   - `draw()` は各レイヤーを順に処理し、`layer.color` があれば `set_line_color`、無ければ基準色へ戻す。`layer.thickness` があれば基準太さに乗算して `set_line_thickness`。その後、`geometry` をアップロードして `LINE_STRIP` 描画。
     - 参照: `src/engine/render/renderer.py:164-199`
   - 直近レイヤー色は `_sticky_color` に保持され、次フレームにレイヤーが無い場合も残色が再適用される（意図的な粘着挙動）。
     - 参照: `src/engine/render/renderer.py:215-219`

6) GUI スナップショットの注入（フレーム毎）
   - ワーカは `draw` 実行直前に `apply_param_snapshot(overrides, t)` を呼び、`SnapshotRuntime` を有効化して「未指定引数のみ」GUI 値を適用可能にする。
     - 参照: `src/engine/ui/parameters/snapshot.py:241`, `src/engine/ui/parameters/snapshot.py:142-199`
   - そのため GUI 変更を反映させるには、パイプライン宣言（`E.pipeline...`）を draw 内で行う必要がある（サンプルはこの前提）。
     - 参照: サンプル `sketch/251107.py:14-33`

**仕様確認（期待どおりか）**
- 非幾何/後勝ち: OK。`style` は plan から除去され、最後の指定のみが `StyledLayer` に反映（`src/effects/style.py:30`, `src/engine/runtime/worker.py:96-120,125-151`）。
- キャッシュ健全性: OK。`style` 除去後の plan で幾何を実体化するため、`style` 変更だけでは形状キャッシュ/Prefix キャッシュに影響しない。
- レイヤー描画: OK。レイヤー配列が来たフレームのみ順描画し、その他は通常経路（残色は `_sticky_color` で維持）。
- GUI 整合: OK。`style.color` は DPG の 0–255 カラーピッカーで編集され、内部は vec3(0..1) として解決（`src/engine/ui/parameters/dpg_window.py:1151-1176,1178-1232`）。

**注意点 / 落とし穴**
- パイプラインの宣言位置:
  - パイプラインを draw の外で一度だけ `build()` すると、その時点の値が固定される。GUI の動的変更は `apply_param_snapshot` によっても既存パイプラインへは反映されないため、毎フレーム宣言（または再構築）する前提をドキュメントで強調すべき。
  - 参照: 宣言→解決タイミング `src/api/effects.py:127-153`, スナップショット注入 `src/engine/ui/parameters/snapshot.py:241`。
- 複数 `style` の扱い:
  - 1 `LazyGeometry` につき「最後の指定のみ」が採用される。中間で色/太さを切り替えてレイヤー分割したい場合は、`LazyGeometry` を配列で返すか、複数のパイプライン適用に分割する必要がある（設計として明確）。
- 値の防御性（色の正規化）:
  - `last_color` の正規化（`_norm_color`）は try/except で保護されていないため、異常値（想定外の型）で `ValueError` がワーカまで伝播し得る。UI 経由では問題になりにくいが、ユーザコードが直接不正値を渡すケースに備え、防御的に握りつぶして無視（`None`）とするのが安全。
  - 提案: `rgba = _norm_color(tuple(last_color))` を try/except でガードし、失敗時は `rgba=None` にフォールバック（`src/engine/runtime/worker.py:118-119,149-150`）。
- 太さの倍率適用:
  - `thickness` はレンダラ初期化時の基準太さ（`line_thickness`）に乗算される。GUI のグローバル太さ変更は用意されていないため（色はある）、必要なら将来の拡張点。
- 粘着色の副作用:
  - レイヤーのないフレームで直前の色が残る仕様（`_sticky_color`）は便利だが、意図せず色が持ち越される可能性がある。常にデフォルト色へ戻したいケースでは、レイヤー無しフレーム前に明示的に `set_line_color` を呼ぶか、設計でオプトアウト設定を検討。

**改善提案（小粒）**
- 防御強化: 色正規化の try/except（上記）。
- ドキュメント追記: 「GUI 変更を反映させるにはパイプラインを draw 内で宣言する」旨を README/architecture に明記。
- 将来拡張: `style.color` の vec4（α）対応とブレンド仕様の整理（ModernGL 側のブレンド状態と整合）。

**確認用チェックリスト（手動）**
- LazyGeometry 1 件に対し `style(color=(1,0,0), thickness=2.0)` 指定 → レイヤー 1 件（赤/2×）で描画されること。
- 同一 `LazyGeometry` に `style(...=青)` → `style(...=赤)` と 2 回指定 → 後者のみ適用されること。
- 
- `(LazyGeometry_without_style, LazyGeometry_with_style)` の配列を返す → 前者は基準色/基準太さ、後者は指定色/倍率で描画されること。

関連資料
- 実装計画: `docs/plans/style_effect.md`

