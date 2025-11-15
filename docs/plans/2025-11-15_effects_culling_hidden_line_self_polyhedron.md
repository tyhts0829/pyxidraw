# culling 自己隠線対応改善計画（単一 polyhedron 含む）

目的

- 単一の 3D 立体（特に `G.polyhedron()` の立方体など）に対しても、向こう側に隠れた線を描画しない「自己隠線消去」を行う。
- 具体的には、立方体を回転させたときに、背面や側面の奥に完全に隠れている辺はプロッタ出力から除去し、前面〜シルエットの辺だけを残す。
- 新しいエフェクトを増やさず、既存の `effects.culling` の内部で自己隠線も処理し、「単一オブジェクト内」と「複数オブジェクト間」の両方を 1 つのエフェクトで扱う。

スコープ

- 対象モジュール:
  - `src/effects/culling.py`（自己隠線ロジックの追加・統合）
  - 必要に応じて `src/util/geom3d_ops.py` を再利用/拡張
- 形状側:
  - `src/shapes/polyhedron.py` の実装は前提情報として利用するが、本タスクでは変更しない（face 情報は Geometry から復元）。
- ドキュメント:
  - `docs/plans/2025-11-15_effects_culling_hidden_line_v2.md`
  - `docs/plans/2025-11-15_effects_culling_3d_geometry.md`
  - 必要に応じて `architecture.md` / `docs/spec/shapes.md` / `docs/spec/effects*.md` との整合を後続タスクで調整。

現状と課題（整理）

- 現状:
  - `effects.culling` は「複数 Geometry 間」の隠線処理を担当し、Z ソート + Shapely による occluder（面）構築 + `difference` で奥側の線を削る実装になっている。
  - `docs/plans/2025-11-15_effects_culling_hidden_line_v2.md`, `2025-11-15_effects_culling_3d_geometry.md` に基づき、face-aware occluder（`_extract_face_polygons` → `_build_occluder_region`）まで実装済み。
  - しかし「1 つの Geometry 内での自己隠線」は対象外であり、`G.polyhedron()`（立方体など）単体では背面の辺もワイヤーフレームとしてすべて残る。
- 課題:
  - 単一立方体に対して「前面だけが見える」表現を行いたい場合、現在は culling だけでは実現できず、ユーザー側で線の選別を行う必要がある。
  - convex polyhedron（正多面体）のようなケースでは、面と辺の対応関係が比較的単純なので、エフェクト側で self-culling を行う余地がある。

要求仕様（ターゲット挙動）

- 対象:
  - 主に `G.polyhedron()` により生成された Geometry（convex polyhedron）を想定する。
  - 他の 3D 形状（sphere/capsule など）は当面スコープ外（将来の拡張候補）。
- 振る舞い:
  - カメラ方向（視線）は Z 軸に沿った正射影を前提とする（現行 culling と同様、`view_dir = (0, 0, -1)` 相当）。
  - Geometry 内の各辺について、その辺に接続する面（face）のうち少なくとも 1 つが前向き（表向き）なら「可視 edge」として残す。
  - すべての接続面が背面（裏向き）の edge は「不可視 edge」とみなし、結果 Geometry から削除する。
  - convex polyhedron では、正しいシルエット + 前面の辺だけが残り、背面だけに属する辺は描画されない。
- 制約・前提:
  - 単一 Geometry 内での自己隠線消去は「convex かつ有限個の平面 face からなる polyhedron」を前提とし、一般の self-intersecting な 3D 線集合は対象外とする。
  - Shapely は「面のポリゴン化」ではなく、必要に応じてシルエット検証などに限定利用し、基本の edge 判定は numpy ベースで行う。

インターフェース方針

- 公開 API は基本的に現行の `culling(geos, *, thickness_mm=0.3, z_metric="mean", front_is_larger_z=False)` を維持する。
- 自己隠線の扱いは、`culling` 内部でのオプションとして実装する:
  - 最小案: convex polyhedron と判定できた Geometry に対しては常に self-culling を有効にする（パラメータ追加なし、挙動改善の破壊的変更を許容）。
  - 代替案（要検討）: `self_culling: bool = True` のような追加パラメータを導入し、GUI 側から on/off できるようにする（API・スタブ更新が必要）。
- 実装レベルでは `_apply_self_culling(g: Geometry, view_dir=(0, 0, -1)) -> Geometry` のような内部ヘルパを導入し、`culling` の処理フローの中で各レイヤーに対して 1 回だけ呼び出す。
- self-culling 適用後の Geometry を、そのレイヤーの可視線抽出と occluder 構築の両方に使うことで、「自己隠線」と「他 Geometry との隠線」を同一ロジックで扱う。

設計方針（アルゴリズム案）

1. Geometry から face（閉じた多角形面）を抽出

- 入力: `Geometry g`（convex polyhedron を想定）
- 手順:
  - `coords, offsets = g.as_arrays(copy=False)` を取得。
  - 各ポリライン `i` について:
    - `s = offsets[i]`, `e = offsets[i+1]` で `loop = coords[s:e]` を切り出す。
    - `||loop[0] - loop[-1]|| < close_eps` であれば「閉じたループ」とみなし、face 候補とする。
    - 頂点数が 4 以上（四角形以上）であることをチェックし、退化ループは除外。
  - 上記は `_extract_face_polygons` のロジックとほぼ共通であり、可能であれば 3D ループ抽出用ヘルパに切り出して再利用する。

2. 各 face の法線と前向き/背向き判定

- 各 face `loop` について:
  - 3D 座標から共分散 or SVD ベースで法線ベクトル `n` を推定する（`_extract_face_polygons` と同様の実装を流用可）。
  - `view_dir` との内積 `dot = n · view_dir` を求める。
    - `dot < -normal_eps` を「前向き」、`dot > normal_eps` を「背面」、`|dot| <= normal_eps` は「ほぼエッジオン」としてとりあえず前向き扱い（または個別検討）。
  - 法線の向き（loop 頂点の回転方向）に依存しないよう、`view_dir` に対して対称な条件になるよう規約を決める。

3. 頂点 ID 付与と edge（辺）の抽出

- 目的: 「どの edge がどの face に属するか」を判定し、edge ごとに incident faces を集約する。
- 手順:
  - まず頂点座標に一意な ID を振る:
    - 各 `loop` 中の各点 `p` について、3D 座標を pos_eps で量子化したキー（例: `(round(x/eps), round(y/eps), round(z/eps))`）を作成。
    - 既存のキーに近い場合は同一頂点 ID を再利用し、新規なら新しい ID を割り当てる。
  - 各 face について、頂点 ID 列 `vids = [v0, v1, ..., v_{k-1}]` を構築する（末尾は先頭と一致している想定）。
  - edge 抽出:
    - 各 face について隣接頂点ペア `(vids[j], vids[(j+1) % k])` を edge 候補とし、順序に依存しない canonical key（例: `tuple(sorted((a, b)))`）を作る。
    - `edge_to_faces[edge_key]` に face のインデックスを追加。

4. edge ごとの可視/不可視判定

- face 判定結果（前向き/背面）と `edge_to_faces` を組み合わせる:
  - 各 edge_key について、その edge に属する face インデックス集合を取得。
  - その中に「前向き face」が 1 つでも含まれていれば、その edge を「可視 edge」とする。
  - すべて背面のみに属する edge は「不可視 edge」として削除する。
- convex polyhedron の場合:
  - 典型的には edge は 2 面に接しており、片方が前向き・もう片方が背面の edge がシルエットとして残る。
  - 両面とも前向きな edge は前面の内部構造だが、convex では通常シルエットとして見える or 表面境界なのでそのまま残して問題ない。

5. 可視 edge から Geometry を再構築

- シンプルさ優先で、可視 edge ごとに「2 頂点だけのポリライン」を生成し、`Geometry.from_lines([...])` でまとめ直す:

  ```python
  visible_lines = []
  for edge_key in visible_edges:
      va, vb = edge_key  # 頂点 ID
      pa = vertex_positions[va]
      pb = vertex_positions[vb]
      visible_lines.append(np.stack([pa, pb], axis=0))
  out = Geometry.from_lines(visible_lines)
  ```

- 元の face ごとのループ構造は失われるが、プロッタ描画の観点では「辺単位のポリライン集合」で十分。
- 極端に短い edge（長さ < length_eps）は数値ノイズとして除外する。

6. フォールバック戦略

- face 抽出に失敗した場合（閉じたループが見つからない / plane 判定 NG）:
  - self-culling は no-op とし、その Geometry は従来どおりの culling（他 Geometry とのみの隠線処理）だけを受ける。
- 2D Shape（Z=0 のみ）の場合:
  - face 抽出は通るものの、自己隠線はそもそも不要なので no-op 扱いでよい（view_dir と法線が常に一定になるため）。

テスト計画

- 追加・更新するテストファイル案:
  - `tests/effects/test_culling_hidden_line.py`（既存テストの拡張）
- 想定テストケース:
  1. 単一立方体の正面ビュー:
     - `G.polyhedron(polygon_index=1)` を使用し、Z 軸方向から見下ろす姿勢（`affine` 等）にして `culling([g])` を適用。
     - 改善前のワイヤーフレーム状態よりも線本数が減少し、背面の辺（Z がより奥側の面に属する edge）が含まれないことを確認。
  2. 単一立方体の斜めビュー:
     - X/Y/Z 回転を組み合わせた姿勢で `culling([g])` を適用し、「手前 3 面 + シルエット」に相当する辺だけが残っていることをざっくり検証（線本数・Z 範囲など）。
  3. 非 polyhedron ジオメトリ:
     - 単純な 2D 矩形などを `culling([g])` に通しても、入出力の `coords` が完全一致する（self-culling が no-op）ことを確認。
  4. 複数オブジェクトとの組み合わせ:
     - 手前立方体（self-culling 対象）と奥側の別 Geometry を用意し、`culling([g_front, g_back])` が「自己隠線 + 他オブジェクトとの隠線」の両方を満たすことを確認。
  5. エッジケース:
     - 非凸形状や自己交差を含む Geometry に対しても例外を投げず、少なくとも処理が最後まで走る（結果が多少期待と異なってもよい）ことを確認。

実装タスク（チェックリスト）

1. 設計/既存コードの整理

- [x] `docs/plans/2025-11-15_effects_culling_hidden_line_v2.md` / `2025-11-15_effects_culling_3d_geometry.md` の内容と今回の self-culling 仕様の関係を整理し、重複/矛盾をメモする。
- [x] `src/effects/culling.py` の `_extract_face_polygons` 実装を読み直し、3D face 抽出ロジックを self-culling 用ヘルパとどう共通化するか方針を決める。
- [x] `src/shapes/polyhedron.py`（特に hexahedron の 6 面表現）が self-culling の前提と矛盾しないことを確認する。

2. culling 内部での自己隠線ロジック追加

- [x] `src/effects/culling.py` に `_apply_self_culling(g: Geometry, view_dir=(0.0, 0.0, -1.0)) -> Geometry` のような内部関数を追加する。（実装では view_dir を内部固定し、Z 変化が小さい場合は no-op とする簡易版とした）
- [x] `culling(...)` のレイヤー処理内で `Geometry` を `ml` に変換する前に `_apply_self_culling` を呼び出し、その結果 `g_self` を以降の visible 線抽出に用いる（occluder 構築は元の Geometry ベースを維持）。
- [x] convex polyhedron と判定できない Geometry（Z 変化が極小 / 頂点数が大きすぎる など）では `_apply_self_culling` が元の `Geometry` をそのまま返すようにし、挙動を現状と同等に保つ。

3. パラメータ/挙動の最終決定

- [x] `self_culling` を常に有効にする案を採用し、convex な 3D polyhedron については内部的に自己隠線を行う（パラメータ追加なし）。
- [ ] パラメータを追加する場合は、`culling.__param_meta__` と API スタブ（`tools/gen_g_stubs.py` 経由）を更新する。（現状は保留）

4. テスト & 品質確認

- [x] `tests/effects/test_culling_hidden_line.py` / `tests/effects/test_culling_hidden_line_3d.py` を更新/拡張し、単一/複数 polyhedron に対する隠線挙動を segment 数ベースで検証する。
- [x] 変更ファイルに対して `ruff check --fix`, `black`, `isort`, `mypy` を実行し、エラー/警告がないことを確認する。
- [x] `pytest -q tests/effects/test_culling_hidden_line.py tests/effects/test_culling_hidden_line_3d.py` を実行し、テストが通ることを確認する。

5. ドキュメント同期

- [ ] `docs/plans/2025-11-15_effects_culling_hidden_line_v2.md` に「self-culling（単一 polyhedron 含む）」が `effects.culling` 内部で扱われることを追記する。
- [ ] `architecture.md` に「polyhedron + culling による 3D 自己隠線 + 他オブジェクト隠線処理パイプライン」の概要を 1〜2 行で反映する（必要であれば別タスク化）。
- [ ] 必要なら `docs/spec/effects_culling.md`（新規）などに culling の仕様を整理し、self-culling 拡張を明記する。

この計画に問題がなければ、このチェックリストに従って `effects.culling` 内部の自己隠線対応実装とテストを進める。***
