# plan: partition の共平面対応＋偶奇領域の安定化（affine 傾き下でも有効）

目的
- `src/effects/partition.py` が「XY 平面限定」ゲートに依存しているため、`affine` で X/Y 回転が入ると無効化（フォールバックで入力をそのまま返す）されてしまう問題を解消する。
- `fill` と同様に「任意の共平面」を検出して XY に整列し、偶奇規則で定まる領域を Voronoi で分割（Shapely あり）、または三角分割で近似（Shapely なし）できるようにする。
- 公開 API（引数/戻り値）は維持し、後段 `fill` との整合（閉ループ出力）を保つ。

非目標
- サイト生成ロジックの高度化（Poisson disk 等）や分割アルゴリズムの追加。
- 出力のフェザリング/ポストプロセスの新規導入。

設計方針
- グローバル共平面フレームの選択 → 全体 XY 整列 → 偶奇領域（外環 XOR 穴） → Voronoi/三角 → 3D へ戻す、の流れに一本化。
- フレーム選択は `fill` と同等の堅牢性（リング優先 → PCA フォールバック）。
- 偶奇領域は Shapely があれば幾何ブールで構築、無い場合は既存のレイキャスト偶奇判定を踏襲。

変更内容（詳細）
1) 共平面フレーム選択の導入
   - 既存の `transform_to_xy_plane` を「全体に一度かける」のではなく、まず各リングに対して適用し、z 残差が許容内なリングの姿勢を採用。
   - 見つからない場合は PCA（SVD）の最小特異ベクトルを法線として回転を生成。
   - 得られた回転で `coords` 全体を XY に整列し、z 残差が絶対/相対閾値（`max(abs, rel*diag)`）以内なら「共平面」とみなす。

2) XY 整列後の領域構築
   - Shapely あり: 各リングを `Polygon` 化し、`symmetric_difference` で偶奇領域（XOR）を構築。
   - Shapely なし: 既存の `_triangulate_polygon_xy` で各リング三角化 → 三角の重心で偶奇判定し、奇数個内包の三角のみ採用。

3) Voronoi または三角分割
   - Shapely あり: 領域の `bounds` 内に `site_count` 個（最大トライアル with seed で再現）点をサンプリングし、Shapely の `voronoi_diagram(..., envelope=region.envelope, edges=False)` → `intersection(region)` → `Polygon` の外周を抽出。
   - Shapely なし: 2) の三角分割結果をそのままセルと見なす（現行フォールバック維持）。

4) 3D への復元
   - XY で得た閉ループを `transform_back` で元の 3D 姿勢へ戻し、`Geometry.from_lines` にまとめる。

5) エッジケース
   - 入力が空/単一リング未満/極端に退化 → 安全フォールバック（入力コピー）
   - 領域が空（XOR の結果が空）→ 入力コピー
   - サイトが確保できない → 代表点 `representative_point()` を 1 個だけ採用

実装タスク（チェックリスト）
- [x] 共平面フレーム選択ヘルパを導入
  - 関数: `_choose_coplanar_frame(coords, offsets) -> (planar: bool, v2d: np.ndarray, R: np.ndarray, z: float)`
  - 実装は `fill` と同等（リング優先 → PCA フォールバック → 全体 XY 整列 → 閾値判定）。
  - 共有ユーティリティ化: `util/geom3d_frame.py`（仮）へ切り出し、`fill/partition` 双方から利用。
- [x] partition の XY 限定ゲートを置換
  - 既存の `_is_planar_xy_all(coords)` 分岐を `_choose_coplanar_frame` へ差し替え。
  - `planar=False` の場合は現行のフォールバック（入力コピー）を維持。
- [x] XY 経路の本体処理を v2d ベースに統一
  - 領域構築（Shapely XOR / 三角の偶奇選別）はすべて `v2d`（XY 整列座標）で行い、最後に `transform_back`。
  - 既存の z=0 埋め込みや閉路化 `_ensure_closed` は温存。
- [x] 依存の最小化と docstring 整理
  - `docstring`: 「XY 限定 → 共平面対応」へ説明を更新。穴の扱い（偶奇）を明記。
  - Shapely の有無で挙動が変わる点はそのまま（skip テストあり）。
- [x] 変更範囲の高速チェック
  - ruff/black/isort/mypy: 変更ファイルのみ。
  - pytest: 既存 `tests/test_effect_partition.py` の smoke に加え、傾き下のケースを追加実行。

テスト計画
- 既存テストの維持
  - `test_voronoi_square_closed`（Shapely 環境）/ `test_partition_then_fill_smoke`/`test_donut_excludes_inner_hole`
- 追加（Shapely 環境のみ実行）
  - 倾斜テキスト: `G.text("9o")` を `affine(angles_rad=(α,β,γ))` で傾け、`partition(site_count=...)` の出力が空でない・全ループ閉路・穴内にセルが入らないこと。
  - 並び依存の排除: `"9o"` と `"o9"` の双方で同条件が満たされる。

リスクと緩和
- フレーム選択の閾値によっては共平面判定がブレる可能性 → `fill` と同一の絶対/相対閾値関数を共有し一貫性を確保。
- Shapely のバージョン差異（`voronoi_diagram` 有無）→ 現行のフォールバック（耳切り）経路を温存。

オプション（要相談）
- [ ] フレーム選択ヘルパを `fill` から `util` へ移設し、重複排除（`fill` 側の差分も同時に整理）。
- [ ] サイト分布を Poisson disk にする軽い改善（別計画）。

完了条件（DoD）
- 倾き下（X/Y 回転あり）でも partition が機能し、空出力にならない。
- 穴（holes）内に Voronoi セル（三角含む）が生成されない（偶奇規則維持）。
- 既存テスト緑、追加テスト緑、`ruff/black/isort/mypy`（変更範囲）緑。
