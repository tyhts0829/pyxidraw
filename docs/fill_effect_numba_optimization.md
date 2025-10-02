# fill エフェクト最適化（Numba 検討と実施チェックリスト）

対象: `src/effects/fill.py`

目的: 出力同等性を保ちつつ、塗りつぶし（lines/cross/dots）の計算コストと Python ループのオーバーヘッドを削減。Numba の適用範囲を見直し、コア処理を nopython 化・配列化する。

---

現状サマリ（2025-09 時点）
- 機能:
  - `lines`/`cross`/`dots` の 3 パターン。
  - XY 共平面は偶奇規則で一括処理（穴保持）。非平面はポリゴンごとに XY へ姿勢合わせ→生成→復元。
- 既存の Numba 化:
  - `util.geom3d_ops.transform_to_xy_plane` / `transform_back` は `@njit(cache=True)`。
  - `find_line_intersections_njit`（水平線×ポリゴン交点）
  - `point_in_polygon_njit`（点の内外）
  - `_point_in_polylines_evenodd_njit`（複数輪郭の偶奇判定）
  - `find_dots_in_polygon`（ドット用グリッド点抽出）
  - `generate_line_intersections_batch`（y ごとの交点リスト）
- 想定ホットスポット/懸念:
  - Python ループが残る箇所
    - `_generate_line_fill_evenodd_multi`: y スキャン×輪郭ループ→交点収集/ソート/区間化を Python で実施。
    - `_generate_dot_fill_evenodd_multi`: グリッド二重ループを Python で回し、内側で Numba 関数呼び出し。
    - 非平面系 `_generate_line_fill`: 生成後の各線分で都度 `transform_back`（多量の小配列生成）。
  - `generate_line_intersections_batch` は `@njit` だが Python の list[(y, ndarray)] を返しており、nopython 非対応の型を含むため設計上ボトルネック化・最適化余地が大きい。
  - ループ内での多数の小配列生成（`np.array`/`np.hstack`）が多い。

---

最適化方針（Numba を“効くところ”へ集中）
- コア計算（スキャンラインの交点計算、偶奇規則の区間化、グリッド内点抽出）を nopython で完結させる。
- 出力が可変長になる箇所は「2 パス（count→allocate→fill）」で前方確保し、配列に直書きする。
- 回転・姿勢復元などの行列適用はベクトル化（可能なら Numba 化済み関数に一括入力）で Python 呼び出し回数を削減。

---

提案する具体的変更（設計スケッチ）

1) スキャンラインの一括区間生成（list 返却の解消）
- 置換案: `scanline_segments_njit(polygon: float32[:, :], y_values: float32[:]) -> (float32[:], float32[:], float32[:], int32[:])`
  - 返り値: `xs0_flat, xs1_flat, ys_flat, seg_offsets`
    - `seg_offsets[k]..seg_offsets[k+1]` が y=k の区間列。
  - 実装: 1 パス目で各 y の交点数→区間数をカウント、合計で flat 長を確定。2 パス目で [x0,x1] を昇順ペアで詰める。
- 既存 `generate_line_intersections_batch` は削除または内部から呼ぶだけのラッパへ。

2) 偶奇規則マルチ輪郭の 2D 区間生成を Numba 化
- 追加: `scanline_segments_evenodd_multi_njit(coords2d: float32[:, :], offsets: int32[:], y_values: float32[:]) -> (xs0, xs1, ys, seg_offsets)`
  - 各 y で全輪郭から交点を収集→ソート→偶奇で [x0,x1] を生成まで nopython で完結。
- Python 側 `_generate_line_fill_evenodd_multi` は
  - y 配列の生成、角度の回転/戻し（行列適用）、z 埋め→`transform_back` or 直接 3D 合成のみに縮小。

3) 偶奇規則ドットのマルチ輪郭版を Numba 化
- 追加: `find_dots_in_polylines_evenodd_njit(coords2d, offsets, x_values, y_values) -> float32[:, :]`
  - 1 パス目カウント→2 パス目 fill。出力は中心点の配列。
- Python 側 `_generate_dot_fill_evenodd_multi` は配列を十字線分へ変換し 3D 化のみ。

4) 非平面ライン塗りのバッチ復元
- `_generate_line_fill` での per-segment `transform_back` 呼び出し回数を削減。
  - アウトライン: 2D で得た全線分をまとめて一つの `float32[:, :, 3]`（形状: [M, 2, 3]）へ格納し、一括で姿勢復元。
  - 必要なら `util.geom3d_ops` に `transform_back_many(vertices3d: float64[:, :, :], R: float64[:, :], z: float64) -> float64[:, :, :]` を追加（内部は現行 `transform_back` のベクトル化）。

5) 小配列生成の削減
- ループ内の `np.array([[...], [...]])`/`np.hstack` を排除し、
  - 2 パスで事前確保した大きい配列へ直接書き込み、最後に `np.split` or offsets で切り出し。

6) 並列化（任意・後段）
- 高密度/大グリッド時は `prange` による y スキャン/グリッド列の並列化を検討（OpenMP/SIMD 環境がある場合のみ）。

---

期待効果（目安）
- 平面XYライン/クロス: 交点収集と偶奇区間化を Numba 側へ移すことで 1.5〜3.0×。
- ドット（平面XY）: グリッド走査の Python ループ排除で 1.3〜2.5×。
- 非平面ライン: per-segment 配列生成と復元の一括化で 1.2〜1.8×。
- いずれも密度が高いほど効果が大きい見込み。

注意/トレードオフ
- Numba の nopython で Python list/tuple-of-arrays を返す設計は不可。必ず配列＋offsets か typed.List のみ（後者は保守性が下がる）。
- JIT 初回コストがあるため、短命ジョブ/低密度では体感差が小さい可能性。
- dtype は `float32`/`int32` を通すと速いが、既存の `transform_*` が `float64` を前提に内部キャストしているため、呼び出し境界での型変換コストとのトレードオフに注意。

---

検証方針（DoD）
- 機能同等性
  - 代表入力で旧実装と比較し、線分端点の最大誤差 ≤ 1e-6（float64 基準）。
  - 偶奇規則で穴の保持が一致すること。
  - 退行ケース（空/2頂点/ゼロ面積、多角形の水平辺など）で安定。
- 性能
  - `tests/perf` の該当ケース（密度大、複数輪郭/多角形）で退行なし、できれば改善。
- チェック
  - 変更ファイルに限定した `ruff/black/isort/mypy/pytest -q` 緑。

---

実施チェックリスト（まずは設計→実装の順で段階導入）
- [x] API 設計確定（返却配列＋offsets の体裁と dtype）
- [x] `scanline_segments_njit` の実装（単一ポリゴン）
- [x] `scanline_segments_evenodd_multi_njit` の実装（複数輪郭）
- [x] `find_dots_in_polylines_evenodd_njit` の実装
- [x] `_generate_line_fill_evenodd_multi` の置換（Python 側は回転/3D 合成のみ）
- [x] `_generate_dot_fill_evenodd_multi` の置換（Python 側は十字生成/3D 合成のみ）
- [x] 非平面 `_generate_line_fill` のバッチ復元対応（`transform_back_many` 追加）
- [x] 既存 `generate_line_intersections_batch` の廃止 or 内部向け置換
- [ ] ベンチと機能回帰テスト、閾値合意（許容誤差/性能目標）
- [ ] `architecture.md` に差分があれば更新（アルゴリズム/制約）

---

補足（現状コードへの短評）
- `generate_line_intersections_batch` は `@njit` だが Python list に `(y, ndarray)` を詰めて返しており、nopython では扱いづらい構造。配列＋offsets へ変更するのが筋。
- 平面XY系では 3D 復元をせず `z0` 固定としており、良い切り分け。Numba 移行のコストが低い。
- 非平面系では `transform_to_xy_plane`/`transform_back` が `@njit` 済みなのは良いが、ループ内の小配列生成が支配的になり得るため、一括処理化が有効。

---

次アクション
- このチェックリストの方針で実装に進めてよいか確認してください。承認後、段階的に PR を分割して進めます（平面XYライン→クロス→ドット→非平面の順を想定）。
