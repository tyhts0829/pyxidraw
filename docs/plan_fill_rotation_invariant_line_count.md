# fill: Z 回転に対して本数を一定にする（パフォーマンス非劣化）

目的

- affine の Z 回転（rz）に対して、fill の生成線本数を安定させる。
- パフォーマンスを落とさず（≈ 現状同等）、既存の見かけ密度の直感も大きく崩さない。
- 破壊的変更は許容（既定挙動の改善）。API 追加は最小限（不要なら追加しない）。

背景（現状と問題）

- 現行実装は「未回転（XY 整列後）の Y スパン」を基準に `spacing` を決める。
  - 共平面経路: `ref_height_global = max(v2d_all[:,1]) - min(v2d_all[:,1])`。
    - 参考: `src/util/geom3d_frame.py:109`
  - `spacing = _spacing_from_height(ref_height_global, density)` を全グループへ配布。
    - 参考: `src/effects/fill.py:359-365`
- 実スキャンの Y スパンは「塗り角 angle を考慮した作業座標」で決まるため、affine の Z 回転により
  参照高さとスキャン範囲の比が変化 → 実生成本数が増減。
- ドキュメンテーションにも「角度により実生成本数は多少変動」と記載があるが、Z 回転に対しては
  変動しないことを目標とする。

設計方針（最小・高速・回転不変）

- 角度 `ang` ごとに「スキャン方向の高さ（scan_height）」で間隔を決める。
  - 定義: `y_proj = x*sin(ang) + y*cos(ang)`（XY での投影）。
  - `scan_height = max(y_proj) - min(y_proj)` を用い、`num_lines = clamp(round(density), 2..MAX)`、
    `spacing = scan_height / num_lines` を採用する。
  - これにより affine の Z 回転に対して本数 ≈ `num_lines` で安定（arange の端数で ±1 することあり）。
- 計算コストは角度ごとに `O(N)` の一次式（投影＋ max/min）で軽く、Numba 交点探索の支配に比べ無視可能。
- グループ間の見かけ密度整合: `scan_height` は「全体 v2d_all」を用いて角度ごとに一度だけ計算し、
  その `spacing` を各グループへ配布（現状同等の整合を維持）。

影響範囲

- 共平面経路（推奨変更・主要）：`src/effects/fill.py` の planar ブロックで `spacing_glob` の決定を
  「`ref_height_global` ベース」から「角度ごとの `scan_height` ベース」へ切替。
- 非共平面フォールバック（任意）:
  - `_generate_line_fill` は「未回転高さ ref_height」を使用しているため、`work_2d` の Y スパンを用いる
    方式へ置換すると一貫性が増す（本数安定）。
  - まずは共平面のみでも主症状は解消（テキスト等は共平面であることが多い）。

実装ステップ（チェックリスト）

- [x] ヘルパ追加: `scan_height = _scan_span_for_angle_xy(coords_2d: np.ndarray, ang: float)`
  - 実装: `y_proj = coords_2d[:,0] * np.sin(ang) + coords_2d[:,1] * np.cos(ang)` → `max-min`。
  - 返り値が `<= 0` の場合は 0.0 を返す（退避処理用）。
- [x] 共平面経路の spacing 決定を差し替え（ファイル: `src/effects/fill.py`）
  - 位置: planar 分岐内、各グループ処理ループ直前（`spacing_glob` を決めている箇所）。
  - 変更: `num_lines = clamp(round(d), 2..MAX_FILL_LINES)` →
    `scan_h = _scan_span_for_angle_xy(v2d_all[:, :2], ang_i)` →
    `spacing_glob = scan_h / num_lines if scan_h > 0 else 0`。
  - `spacing_glob <= 0` の場合はその角度の生成をスキップ。
- [ ] 非共平面フォールバックの単一ポリゴン経路（任意：一貫性向上）
  - 位置: `_generate_line_fill`
  - 変更: 既存 `ref_height` の代わりに、作業座標 `work_2d` の `min_y/max_y` から
    `scan_height = max_y - min_y` を使用し `spacing = scan_height / num_lines` へ。
  - `num_lines` は同様に clamp(round(density), 2..MAX)。
- [x] ドキュメンテーション更新
  - `src/effects/fill.py` の docstring/Notes を「Z 回転に対して本数は概ね不変（±1 の誤差あり）」に更新。
- [x] 変更局所の ruff/black/isort/mypy を実行（編集ファイル優先の高速ループ）。
- [x] テスト追加（smoke で十分）
  - 例: 長方形/テキストを `affine(rz=θ)` で回したとき、`fill(density=K, angle_sets=1)` の
    生成線数が θ=0 と θ=π/4 で同程度（±1 以内）になること。
  - グループ化経路でも同様の安定性を確認（`build_evenodd_groups` 経由）。
- [ ] パフォーマンス確認
  - 代表ケース（文字列/多輪郭）での処理時間が誤差範囲であることを確認。

検証の観点

- 正しさ:
  - Z 回転 θ を複数（例: 0, π/6, π/4, π/3）で比較し、各 θ で線本数 ≈ `round(density)` を保つ。
  - `angle_sets > 1` の場合、各角度 i で独立に `scan_height_i` を用いるため、各方向の本数が一定。
- 性能:
  - 角度ごとに 1 回の投影＋ max/min（`O(N)`）追加。交点探索（Numba）コスト支配に比べ軽微。
- 後方互換:
  - 見かけ密度（線間隔）は角度毎に若干調整されるが、視覚上の違和感は小さい。
  - 本数の安定性を優先する要件に合致。

既知の注意点 / トレードオフ

- `np.arange(min_y, max_y, spacing)` の性質上、端数で ±1 本ぶれることがある。
  - 必要なら `num_steps = num_lines` に合わせて開始位置を調整（`min_y + 0.5*spacing` など）する
    微調整も検討可能（今回は簡潔さ優先で据え置き）。
- `scan_height` が極小（ほぼ線状）だと `spacing → 0` になり得るため、その角度はスキップ。

オプション検討（保留）

- API フラグ追加: `line_count_mode={"ref_height"|"scan_height"}`。
  - 既定を `scan_height` に切替、旧挙動を保持したい場合のみ `ref_height` を指定可能にする。
  - 現状は「破壊的変更許容」の方針に合わせて追加不要と判断。

確認したいこと（要回答）

1. 今回は共平面経路のみで十分か、非共平面フォールバックにも適用するか。：共平面経路だけ（済）。
2. arange による ±1 本のブレは許容で良いか（許容なら簡潔・高速、厳密化も可能）。：多少のブレは OK（済）。
3. `density → num_lines` の丸め・クランプ（2..MAX）は現状のままで良いか。：はい（済）。

完了条件

- 編集ファイル（`src/effects/fill.py`）に対して ruff/black/isort/mypy/対象テストが緑。
- 共平面（テキスト/多輪郭）で Z 回転に対し本数が安定することを確認。
  - スモークテスト（正方形）：0° と 45° の本数差 ≤ 1（済）。

参照

- `src/effects/fill.py:356-365`（spacing 決定箇所）
- `src/util/geom3d_frame.py:109`（現行の参照高さの算出）
- `src/effects/fill.py:214`（スキャン y 値列の生成）
