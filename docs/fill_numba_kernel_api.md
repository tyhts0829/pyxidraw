# fill エフェクト Numba カーネル API（確定版）

目的: `fill` の塗りつぶし生成を nopython で高速化するための内部 API を確定する。出力はすべて「配列＋offsets」による可変長表現とし、Python 側は回転・3D 合成のみを担う。

方針と型規約
- dtype
  - 2D 計算（交点・偶奇・グリッド）: `float32` / `int32`
  - 3D 復元（姿勢復元）: `float64`（`util.geom3d_ops` の実装に合わせる）。
- 配列の前提
  - C-contiguous、要素数 > 0 のときは `shape[1]` が期待次元を満たす。
  - ポリゴンは「閉ループ」を想定（先頭点の重複は不要）。最終辺は `i=(n-1)`→`(i+1)%n=0` で結ぶ。
- 境界条件（水平線×辺の交差）
  - 交差判定は半開区間で実装し、頂点の二重カウントを避ける。
    - エッジ交差: `(y1 <= y < y2) or (y2 <= y < y1)`
- ゼロ長区間
  - [x0,x1] で `x1-x0 <= EPS` の区間が出る場合がある（接線/頂点一致）。必要なら呼び出し側で除去。
  - 既定 EPS は `1e-9` を推奨。

---

API 一覧

1) 単一ポリゴンのスキャンライン区間抽出
- 関数
  - `scanline_segments_njit(polygon: float32[:, :], y_values: float32[:]) -> (float32[:], float32[:], float32[:], int32[:])`
- 入力
  - `polygon`: 2D 座標（N,2）。閉ループ（最後と最初が辺で結ばれる）を想定。
  - `y_values`: スキャンする y 値配列。
- 出力
  - `xs0, xs1, ys`: フラット化した区間列。i 番目の区間は `(xs0[i], ys[i])`→`(xs1[i], ys[i])`。
  - `seg_offsets`: `len(y_values)+1`。`seg_offsets[k]:seg_offsets[k+1]` が `y_values[k]` に対応。
- 仕様
  - 各 y で水平線とポリゴンの交点を収集→昇順ソート→偶奇ペアで [x0,x1] に変換。
  - 2 パス（count→fill）で前方確保。

2) 複数輪郭（偶奇規則）のスキャンライン区間抽出
- 関数
  - `scanline_segments_evenodd_multi_njit(coords_2d: float32[:, :], offsets: int32[:], y_values: float32[:]) -> (float32[:], float32[:], float32[:], int32[:])`
- 入力
  - `coords_2d`: 全輪郭を連結した 2D 座標（N,2）。
  - `offsets`: M+1。i 本目の輪郭は `coords_2d[offsets[i]:offsets[i+1]]`。
  - `y_values`: スキャンする y 値配列。
- 出力/仕様
  - 単一ポリゴン版と同じ形式。交点はすべての輪郭から収集し、ソート・偶奇ペア化まで nopython 内で完結。

3) 複数輪郭（偶奇規則）のドット抽出
- 関数
  - `find_dots_in_polylines_evenodd_njit(coords_2d: float32[:, :], offsets: int32[:], x_values: float32[:], y_values: float32[:]) -> float32[:, :]`
- 入力
  - `coords_2d`, `offsets`: 上記と同じ。
  - `x_values`, `y_values`: グリッド座標（走査対象）。
- 出力
  - `centers`: (K,2)。偶奇規則で内部と判定されたグリッド中心点。
- 仕様
  - 2 パス（count→fill）。内外判定は `_point_in_polylines_evenodd_njit` を利用。

4) 3D への姿勢復元（大量の線分を一括処理）
- 関数
  - `transform_back_many(segments: float64[:, :, :], rotation_matrix: float64[:, :], z_offset: float64) -> float64[:, :, :]`
- 入力
  - `segments`: (M,2,3)。XY 平面基準で z=0 の線分配列。
  - `rotation_matrix`, `z_offset`: `transform_to_xy_plane` が返す値。
- 出力
  - `segments`: (M,2,3)。元の 3D 姿勢に復元した線分配列。

---

使用上の注意
- dtype の境界
  - カーネルは `float32`/`int32` を前提とするため、Python 側で `astype` を最小限に行う。
  - 3D 復元は `float64` を使用（`util.geom3d_ops` の既存実装と整合）。
- 回転角
  - 角度付きライン生成では、回転は Python 側で一括適用してからカーネルに渡す（または戻しのみ適用）。
- 安定性
  - 水平辺や鋭い凹多角形でも、半開区間の交差規約により二重カウントを避ける。
  - ゼロ長区間は出力され得る。必要に応じて `x2-x1 > EPS` で除去。

---

互換性と移行
- 旧 `generate_line_intersections_batch(polygon, y_values) -> list[(y, xs)]` は廃止。
  - nopython で扱えない list-of-arrays を排除し、配列＋offsets に一本化。
- 既存の Python 実装からの移行では、出力のラグド構造は `seg_offsets` で切り出す（`xs0[offsets[i]:offsets[i+1]]` など）。

---

テスト観点（API 仕様の受け入れ条件）
- 同一入力に対し、旧実装のスキャンライン結果と端点誤差 ≤ 1e-6（float64 換算）。
- 偶奇規則で穴の保持が一致。
- 退行ケース: 空/2頂点/水平辺含み/ゼロ面積で安定に空出力。
- dtype チェック: 出力配列の dtype/shape が仕様通り。

