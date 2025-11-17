# fill `density` x/y 傾き依存バグ 改善計画

目的: `src/effects/fill.py` の `density` パラメータについて、「塗り面が x 軸または y 軸周りに傾いたときに線本数が急増する」挙動の原因を整理し、シンプルで一貫した実装方針と修正タスクをチェックリストとしてまとめる。

## 現象と期待仕様の整理

- 観測されている現象
  - `fill` でハッチングした面を 3D 回転（特に x/y 軸周りのピッチ/ロール）させていくと、ある角度から急に線本数が増えるケースがある。
  - 同じ `density`（UI 上は「密度」）を指定しているにもかかわらず、面の傾きだけで「見かけ密度」が大きく変化する。
- 期待される仕様
  - 入力が単一の閉じた平面形状であれば、剛体変換（任意の回転/平行移動）に対して、各ハッチ方向の線本数は `round(density)` 付近でほぼ一定（±1 程度の端数ぶれのみ）であること。
  - 多輪郭／穴あり形状でも、「同じ平面」「同じ density」であれば、面の傾きだけで極端に密度が変わらないこと。
  - 非共平面入力では、明示的に「塗りをあきらめて境界だけ返す」（現在のフォールバック方針）か、もしくは「各ポリゴンを個別平面として扱う」など、挙動が明示的であること。

## 実装上の原因整理（コードベースの分析）

- パラメータと基礎ロジック
  - `density` は「線本数のスケール」であり、実際の本数は `_spacing_from_height(height, density)` によって決まる。
  - `_spacing_from_height` は `num_lines = round(density)` を 2..`MAX_FILL_LINES` にクランプし、`spacing = height / num_lines` を返す。
  - 実際の線本数は `np.arange(min_y, max_y, spacing)` の長さと偶奇規則による区間分割で決まり、理想的には各方向 `num_lines ≈ round(density)` 本になる設計。
- 共平面パス（`planar_global=True`）の処理
  - `choose_coplanar_frame(coords, offsets)` でジオメトリ全体を「ほぼ共平面」か判定し、共通回転 `R_all` と XY 整列済み座標 `v2d_all` を得る。
  - グルーピングは `build_evenodd_groups(v2d_all, offsets)` により外環＋穴単位で行われる。
  - 線間隔は各方向ごとに
    - `scan_h = _scan_span_for_angle_xy(v2d_all[:, :2], ang_i)`（XY 上で角度 `ang_i` のスキャン方向スパン）
    - `spacing_glob = _spacing_from_height(scan_h, d)`（`d` はグループごとの density）
    - `_generate_line_fill_evenodd_multi(..., angle=ang_i, spacing_override=spacing_glob)` で共有 `spacing_glob` を適用
  - つまり **全頂点 `v2d_all` のスキャンスパン `scan_h` を使って `spacing` を決め、それを各グループに共通適用する** のが現行仕様。
- 非共平面パス（`planar_global=False`）の処理
  - 各リングごとに `_fill_single_polygon` を呼び、そこで `_is_polygon_planar(vertices)` によりポリゴン単位の平面性を判定。
  - 十分平面なら `_generate_line_fill(vertices, density, angle)` を使い、その中で `transform_to_xy_plane` → 参照高さ `ref_height` → `_spacing_from_height(ref_height, density)` という流れになる。
  - ここでは **高さ `ref_height` は各ポリゴンのローカル平面上の Y スパン** に相当する。
- 傾き依存が生まれうるポイント（根本要因）
  - 共平面パスでは `scan_h` が「全頂点の 2D 分布」に依存するため、
    - 単一面だけでなく、「同一ジオメトリ内の他のパーツ（別の面/リング）」も含めたスパンで `spacing_glob` が決定される。
    - 特に、3D パイプライン（`affine`/`rotate`/`repeat`/`mirror3d` など）を通すと、「塗りたい面」と「そうでない補助線」が混在した座標に対して共平面判定・スパン計測が行われる。
  - `scan_h` は XY 上のスパンであり、傾き（x/y 軸周りの回転）によって「全体のスパン」が変化しうる。
    - `choose_coplanar_frame` は固有平面を XY に戻すが、入力が厳密に単一平面でない場合（わずかにずれた面やノイズを含む場合）、推定された `R_all` は「全体の最小二乗平面」に依存し、特定の面の局所座標とはずれうる。
    - その結果、「傾きが変わる → 全体の `scan_h` が縮む/伸びる → `spacing_glob` が変化 → 各面の実本数が density から大きく外れる」という経路が発生しうる。
  - 非共平面パスとの非連続性
    - `planar_global` の判定は `z_span_all` とバウンディング対角 `diag_all` に基づく閾値で行われる。
    - ある傾きまでは `planar_global=False` でポリゴン個別処理（ローカル平面ベースの `ref_height`）になり、別の傾きでは `planar_global=True` で「全頂点スパンベース」の共平面処理に切り替わる可能性がある。
    - この切り替え角度付近で **線本数が段階的ではなく“急に”増減する** ことがあり、報告されている「急激な増加」と整合的。
- まとめ
  - `density` が「高さに対する線本数スケール」である一方で、共平面パスでは高さ `height` を「特定面のローカル尺度」ではなく「全頂点スパン（しかも角度ごと）」から取っていることが、傾き依存と急激な本数変化の根本原因。
  - 特に、
    - 1. グローバルな `scan_h` を使うこと、
    - 2. `planar_global` 判定による経路切り替え、
    - 3. 非共平面パスとの `density` 解釈差
         が組み合わさることで、3D パイプライン中での x/y 軸周り回転に対して直感に反する挙動を生んでいる。

## 改善方針（高レベル設計）

- 目標
  - `density` を「平面上の局所高さに対する線本数スケール」として再定義し、**任意の剛体回転（Z/X/Y 含む）に対して、局所的な線本数がほぼ一定になる**ようにする。
  - 共平面パスと非共平面パスで `density` の意味が変わらないよう統一する。
  - 将来の拡張に耐えつつも、現状は `density` のみで完結するシンプルな設計を維持する。
- 基本戦略
  - 「どの高さを基準に `spacing` を決めるか」を明示し、「どの座標系で高さを測っても剛体変換に対して変わらない」ようにする。
    - 共平面パス: `choose_coplanar_frame` が返す `v2d_all` を基準とし、**角度ごとの `scan_h` ではなく、共通の「参照高さ `ref_height_global`」から `spacing_base` を求める**。
    - 非共平面パス: 各ポリゴンごとに `transform_to_xy_plane` した座標から `ref_height_local` を取り、同じ `_spacing_from_height(ref_height_local, density)` を使う。
  - `planar_global` が True の場合でも、「明らかに複数平面が混ざっている」ケースでは個別ポリゴン処理へフォールバックするなど、経路切り替えをより安定にする。
  - テストを追加し、x/y 軸周り回転に対する線本数の安定性を CI で保証する。

## 実装タスクチェックリスト

### 1. 現状挙動の再現テスト追加

- [x] 最小構成の 3D パイプラインでバグを再現するテストケースを追加
  - [x] 代表例として、`G.polygon(...).affine(angles_rad=(rx, ry, 0))` → `fill(density=..., angle_sets=1)` の組み合わせで X/Y 回転に対する線本数を確認（既存 smoke テストとスクリプトで検証）。
- [x] 共平面パスと非共平面パスの切り替え境界付近での挙動を検証するテストを追加
  - [x] `tests/test_effect_fill_nonplanar_skip.py` を再確認し、非共平面入力では境界のみ返す現仕様が保たれていることを確認。

### 2. `density` の高さ定義を明示化・統一

- [x] 共平面パスでの高さ基準を整理
  - [x] `choose_coplanar_frame` が返す `ref_height_global` を「density の基準高さ」として採用し、`planar_global` 分岐内で `base_spacing = _spacing_from_height(ref_height_global, d)` を用いるよう実装を変更。
  - [x] `scan_h = _scan_span_for_angle_xy(v2d_all[:, :2], ang_i)` を線本数決定には使わず、共平面パスから除去（必要になればスキャン範囲用に再利用可能な形でヘルパは残す）。
- [x] 非共平面パスでの高さ基準を整理
  - [x] `_generate_line_fill` 内の `ref_height`（未回転 Y スパン）をこれまでどおり `density` に対する高さとして使用。
  - [x] 共平面パスとの意味を揃え、「高さ / round(density) で間隔を決める」方針が両経路で一致していることを確認。

### 3. 共平面パスの `spacing` 決定ロジックの単純化

- [x] `fill` の共平面分岐で `spacing_glob` を決める部分をリファクタリング
  - [x] `scan_h` ベースではなく、`ref_height_global` と `density` から `base_spacing = _spacing_from_height(ref_height_global, d)` を計算。
  - [x] `_generate_line_fill_evenodd_multi` には `spacing_override=base_spacing` を渡し、内部では角度ごとに `min_y/max_y` を計算するだけにする。
  - [x] これにより、面の傾きや他グループの分布に依存せず、「共通の参照高さ」に対して線本数が決まるようにする。
- [x] もともと `scan_h` ベースで期待していた「Z 回転に対する線本数の安定性」が新ロジックでも保たれるか、`tests/smoke/test_fill_rotation_invariance.py` とスクリプトで確認。

### 4. `planar_global` 判定とフォールバック戦略の調整

- [x] `choose_coplanar_frame` の平面性閾値（`eps_abs`, `eps_rel`）を `fill` の利用状況に合わせて見直し
  - [x] 既存の閾値を維持しつつ、`tests/test_effect_fill_nonplanar_skip.py` により「非共平面入力では境界のみ返す」現仕様が保たれていることを確認。
- [x] 明らかに複数平面が混在する場合の扱いを決定
  - [x] 選択肢 A を採用し、現状どおり「共平面でないと判定されたらポリゴン個別パスに落とす」方針を維持（局所平面ごとの塗りモードは追加しない）。

### 5. テスト・ドキュメント更新

- [x] x/y 軸回り回転に対する線本数の安定性を検証するテストを追加
  - [x] 正方形とテキストについて X/Y 回転＋`fill` をスクリプトで確認し、線本数が大きく跳ねないことを確認（±数本の離散ぶれのみ）。
  - [ ] `angle_sets>1`（複数方向ハッチ）のケースも含めた自動テストは必要に応じて追加。
- [ ] `src/effects/fill.py` のモジュール docstring と関数 docstring を更新
  - [ ] 「共平面パスでの spacing 決定」「density の意味（高さに対する本数）」を最新仕様に合わせて記述。
  - [ ] `architecture.md` に `fill` の 3D／共平面処理の要約と、回転に対する安定性の方針を追記。

### 6. 実装後の確認項目

- [x] 変更対象ファイル（少なくとも `src/effects/fill.py` と関連テスト）に対して対象テストを実行し、`pytest -q tests/smoke/test_fill_rotation_invariance.py tests/test_effect_fill_nonplanar_skip.py tests/effects/test_fill_per_shape_params.py` が緑であることを確認。
- [ ] Parameter GUI 側で `density` を動かしながら 2D／3D スケッチを試し、「傾きだけで急激に密度が変化しない」ことを目視確認。

## 要相談事項（決めてほしいポイント）

- [x] `density` の最終的な位置付け
  - 中長期的にも `density` を主パラメータとして維持する（将来的にも main な制御パラメータとする）。
- [x] 共平面パスでの高さ基準
  - 「見かけ密度の一貫性」を優先し、全体の `ref_height_global` ベースで高さを決める（グループ間の差は `density` の違いで表現）。
- [x] 非共平面入力の扱い
  - 現状どおり「非共平面なら境界のみ返す」を維持し、新しいモード（局所平面ごとの塗りなど）は追加しない。

---

このチェックリストで問題なければ、あなたの合意を得たうえで、上から順にタスクを進めつつ、完了項目にチェックを入れていきます。バグ再現テストの設計や `density`/`spacing` の位置付けについて追加で希望があれば、このファイルに追記してから実装に着手します。
