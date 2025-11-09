% `clip` 改善計画: 非共平面アウトラインでも「XY 投影マスク」で確実に効かせる

目的
- outline（閉曲線マスク）がどんな姿勢/位置でも、傾きや非共平面に関わらず「XY 平面に投影した閉曲面」としてクリップを成立させる。
- 既存の“共平面時の高精度クリップ（Shapely/偶奇規則）”は維持しつつ、非共平面時は投影フォールバックで安定動作。

非目標
- 3D 体積的なクリップ（厚みや体積を持つ切断）。
- 3D で複数平面を跨る厳密クリップ（全点の厳密な最近傍写像）。
- スタイルやレンダリング属性の変更（`style` で扱う）。

要件（改善点）
- outline が x 軸回りに回転して少し傾いていてもクリップが機能する。
- outline は「ワールド XY 平面への投影」結果の閉曲線群として解釈し、偶奇規則（外環+穴）を尊重。
- 対象ポリラインは XY（2D）で判定・切断し、3D の出力を自然に復元する（各元線分の 3D 補間）。
- 共平面である場合は既存の経路（`choose_coplanar_frame` + Shapely 交差）を優先利用して後方互換を確保。

設計方針（概要）
1) モード追加（後方互換）
   - 既定: `auto`。共平面なら現行ロジック、非共平面なら `project_xy` へフォールバック。
   - 将来拡張: `project_mask_plane`（outline の局所平面へ投影）も検討可能だが、初回は `project_xy` を実装対象に限定。
   - パラメータは実値主義に従い boolean を採用（enum は避ける）。
     - `use_projection_fallback: bool = True`（非共平面時の投影を許可）
     - `projection_use_world_xy: bool = True`（True: XY 投影、False: 将来の mask 平面投影に備えた切替）

2) project_xy パス（非共平面フォールバックの新実装）
   - マスク:
     - outline の各リング 3D → XY へ投影（`ring3d[:, :2]`）、`ensure_closed`。
     - 偶奇規則は既存どおり（外環/穴混在 OK）。Shapely があってもこのパスでは「自前実装」を使い、3D 復元を安定化。
   - 対象:
     - 各ポリラインを“辺ごと”に XY へ投影（端点 a,b の 2D）。
     - 各線分 AB について、全マスクリングの辺と 2D で交差探索→ t 値（0..1）を収集→ソート→区間化。
     - 各区間の中点の包含判定（偶奇）で inside/outside を決定。
     - 採用区間 [t0,t1] は 3D へ復元: `A3D + t*(B3D-A3D)` で端点を線形補間（Z を自然に維持）。
   - 出力: 区間ごとに 3D 線分を構築して返す。`draw_outline=True` なら元 3D outline を連結。

3) 共平面時の既存経路は維持
   - `choose_coplanar_frame` で planar=True の場合は従来どおり Shapely を優先（高精度/高速）。
   - planar=False かつ `use_projection_fallback=True` のときだけ `project_xy` へ移行。

4) 数値安定性・性能
   - 交差計算は Numba 実装（既存 `_segment_intersections_with_polygon_edges` ベース）を流用・強化。
   - マスク辺/対象辺の AABB で早期除外。
   - 交差 t の重複/近接は丸め（epsilon=1e-9）でユニーク化。
   - 中点判定は `util.polygon_grouping.point_in_polygon_njit` を利用。

5) 復元ポリシー（3D）
   - project_xy パスでは、切断区間は元の単一線分上の割合 t に基づき 3D 端点を線形補間（Z を自然維持）。
   - 複数セグメントのポリラインは、各セグメント単位で同処理（出力は独立した線分群）。

公開 API 変更（案）
- 既存シグネチャに boolean を追加（既定は安全な後方互換）:
  - `use_projection_fallback: bool = True`
  - `projection_use_world_xy: bool = True`  # 初回は True 固定同然（将来の mask 平面用スイッチ）
- `__param_meta__` には `boolean` を追加。GUI 表示は任意（通常は非表示でも良い）。

実装ステップ（チェックリスト）

1) 事前整理/抽出
- [x] 既存クリップ内の「非共平面→簡易フォールバック」部分を関数抽出（XY ベースの汎用 2D クリップ関数）。
- [x] outline→XY 投影→偶奇判定用データ構造（リング配列/AABB など）の前処理をユーティリティ化。

2) project_xy パスの実装
- [x] `planar=False and use_projection_fallback=True` で `project_xy` へ分岐。
- [x] マスクリング: 3D→XY 投影・閉路化。
- [x] 対象線分: セグメントごとに交差 t 収集→区間化→中点偶奇→採用区間の 3D 復元（A3D/B3D 線形補間）。
- [x] `draw_inside/draw_outside/draw_outline` の論理分岐を維持。

3) パラメータ/メタ
- [x] `clip` シグネチャに `use_projection_fallback`/`projection_use_world_xy` を追加。
- [x] `__param_meta__` に boolean を追加（量子化不要）。

4) テスト
- [ ] 傾き小（x 回り 5–10°）の outline でも grid を正しく内側クリップ。
- [ ] 傾き大（30–45°）でも XY 投影として期待どおりのクリップ形状。
- [ ] 複数リング（外環+穴）で偶奇規則が維持される。
- [ ] 非共平面かつ `use_projection_fallback=False` の場合は従来どおり no-op。
- [ ] 共平面時は従来経路と一致（許容誤差内）。

5) ドキュメント/整合
- [ ] `docs/plans/251109_clip_effect.md` に「非共平面時の投影フォールバック」追記。
- [ ] `architecture.md` のエフェクト仕様メモを更新（共平面/投影の二段階）。

備考/トレードオフ
- Shapely を project_xy パスでも使うと 2D 区間→元 3D 線への写像が煩雑になるため、本パスは「自前セグメント分割」を採用（Z を自然に復元可能）。
- target が XY 以外の平面にある場合でも、「XY 投影としてのクリップ」になる点は仕様（要件に一致）。厳密 3D クリップは非目標。

リスクと回避
- 交差の数値誤差→ t 丸め/境界包含の半開区間ルール（既存実装と同等）で安定化。
- パフォーマンス→ AABB 早期除外と Numba 関数の再利用で軽量化。

スケジュール（概算）
- 実装/局所リファクタ: 0.5–1.0 日
- テスト/微調整: 0.5 日
