# culling エフェクト 3D Geometry 対応改善計画（face-aware occluder）

目的

- `G.polyhedron()` など 3D 的な Geometry を用いたシーンでも、手前オブジェクトが奥オブジェクトを自然に隠すようにする。
- 特に、手前の立方体を回転させても奥の立方体が「透けて」見えることがないよう、隠線領域の構築を 3D 向けに安定させる。
- 既存の API（`E.pipeline.culling(geos=[...])`）を維持しつつ、内部ロジックを 3D Geometry 前提でも破綻しにくい形に拡張する。

現状の問題点（3D Geometry に対する破綻パターン）

- 現行 culling は「1 Geometry ≒ 1 枚のレイヤー（ほぼ一定 Z の平面内）」を前提にしている。
  - XY 平面への投影線から Shapely の `polygonize` で面を構成し、その面を「隠線領域」として使用する 2.5D 設計。
- `G.polyhedron()`（立方体）のような 3D Geometry では、この前提が崩れている:
  - 1 つの Geometry 内に「前面・背面・側面など複数平面」の輪郭線がすべて含まれており、Z も `[-d, d]` と大きくばらつく。
  - これらをまとめて `polygonize` に渡すと、回転角度によっては自己交差・重なりが増え、「シルエット面」がうまく構成できない。
  - 結果として occluder が穴だらけになり、奥の立方体の線が「透けて」見えたり、逆に意図しない部分だけ消えたりする。
- 一方で、2D の正方形 Geometry は「本当に 1 枚の平面レイヤー」なので、既存ロジックで安定して動作する。

要求仕様（3D 対応時のターゲット挙動）

- 3D 的な Geometry（立方体・多面体など）について:
  - 手前オブジェクトの「スクリーン空間シルエット（前向きの面の投影）」を隠線領域として扱う。
  - このシルエット領域内にある奥オブジェクトの線は原則すべて削除する（少なくとも現状のような大きな透けは起こらない）。
- 前提:
  - カメラは Z 軸方向からの正射影（現行パイプラインと同様の 2.5D 前提）。
  - 自己隠線（同一立方体内での隠線消去）は「できれば改善したいが、優先度は奥オブジェクト間の隠線 ＞ 自己隠線」とする。
  - 不特定の 3D Geometry 全般を完全に扱うのではなく、`G.polyhedron` を含む「閉じた面を持つ Shape」が主なターゲット。

改善方針（face-aware occluder の導入）

- 各 Geometry について、「前向きの面」を抽出し、その投影ポリゴンを occluder として利用する:
  - Geometry 内のポリラインから「閉じたループ」を検出し、各ループを 1 枚の面候補（face）とみなす。
  - 各 face について 3D 平面を推定し、法線ベクトル `n` を求める。
  - カメラ方向（例: +Z から -Z への視線）との内積により前向き/後向きを判定し、前向き face のみ occluder の候補にする。
  - face の XY 投影を Shapely Polygon として構築し、これらの union をその Geometry の「シルエット面」として使う。
- 既存の polygonize ベース occluder はフォールバックとして残し、face 抽出に失敗した Geometry では従来ロジックを利用する。

設計詳細案

### 1. Geometry から face（面候補）を抽出する

- 入力: `Geometry`（coords: (N,3), offsets: (M+1,)）
- 手順（face 候補の検出）:
  - 各ポリライン `i` について:
    - `start = coords[offsets[i]]`, `end = coords[offsets[i+1]-1]` を取り、`||start - end|| < eps` なら「閉じたループ」とみなす。
    - 閉じたループのみを face 候補として扱う（線分だけのエッジは面にならない）。
- 手順（平面フィットと法線推定）:
  - 各閉ループ `loop` について:
    - 3D 座標 `loop[:, 0:3]` から平均を引き、共分散行列 → 固有値分解で最小固有値に対応する固有ベクトルを法線 `n` として求める。
    - 各点の平面からの距離が閾値以下（例: `max_dist <= plane_eps`）なら「ほぼ平面」とみなす。
    - loop の XY 投影が十分な面積を持つか（`area >= area_eps`）を Shapely Polygon で確認し、極端に細長い/退化したループを除外する。
- 前向き/後向きの判定:
  - カメラの視線方向を `view_dir = (0, 0, -1)` などで固定し（Z 正方向から原点を見る想定 等）、`dot(n, view_dir)` の符号で向きを判定。
  - `dot(n, view_dir) > 0` の face を「前向き」として occluder 候補に採用。
  - 逆向きの面は occluder には入れない（背面は隠線には寄与しない前提）。

### 2. face ベース occluder の構築

- 前向き face について:
  - XY 投影を Shapely Polygon として構築:

    ```python
    from shapely.geometry import Polygon

    poly = Polygon(loop[:, :2])
    ```

  - 複数 face を `unary_union` でまとめ、1 つの Polygon/MultiPolygon にまとめる。
  - `thickness_mm` は境界マージンとして扱い、小さな buffer（例: `radius = thickness_mm * 0.5`）で膨らませることで境界近傍の数値誤差を吸収する。
- face が 1 枚も取れない場合:
  - 現行の `_build_occluder_region`（polygonize ベース）にフォールバックする。
  - それすら難しい／thickness が 0 の場合は occluder なし（現状どおり）。

### 3. culling 本体への統合（3D-aware occluder パス）

- 現状:
  - `_build_occluder_region(ml, thickness_mm)` が `MultiLineString` から occluder（Polygon/MultiPolygon）を構築。
- 3D 対応案:
  - `_build_occluder_region` を拡張し、以下の順で試行する:
    1. face-aware パス:
       - Geometry/lines から前向き face を抽出し、Polygon union で occluder を作る。
       - 成功したらそれを返す。
    2. polygonize パス（現行）:
       - `polygonize(ml)` → union → buffer で occluder 作成。
    3. buffer fallback:
       - どちらもダメな場合は `ml.buffer(thickness_mm * 0.5)` にフォールバック。
  - 現行の `occluder_region` union ロジック（手前→奥で順次 union）と `difference(occluder_region)` による隠線処理自体は維持。
- 対象 Geometry の判定:
  - 全 Geometry に対して一律 face-aware を試みるが、`plane_eps / area_eps` によって 2D 図形は自然に「1 枚の face」として扱われる。
  - 特別なフラグ（「これは 3D」など）は付けず、形状から推定する。

### 4. 制約と非目標

- 非目標:
  - 任意形状の完全な隠線/隠面消去（一般 3D レンダリングエンジンと同等の機能）。
  - 自己隠線の完全対応（同じ立方体内の奥の辺を全て消す等）。
- 制約:
  - face 検出は「閉じたループ」を前提とするため、線だけのワイヤー形状では従来と同程度の挙動に留まる。
  - 立方体以外の polyhedron でも「面ごとの閉ループ」がある程度きれいに定義されていることが前提。

テスト計画（3D 向け）

- 立方体×2 の隠線:
  - `front = G.polyhedron().scale(...).translate(..., z=0)`、`back = G.polyhedron().scale(...).translate(..., z>0)` を用意。
  - `E.pipeline.affine()` 等で適当な回転を加えた上で、`E.pipeline.culling(geos=[front_geo, back_geo])` を適用。
  - いくつかの代表角度（例: 回転 0°, 30°, 45°, 60°）で:
    - `back` 側の Geometry の線分数が、culling 前より明らかに減少していること。
    - 少なくとも「前立方体のシルエット内」にある線がほぼ消えていること。
- 自己交差・背面の影響:
  - 1 個の立方体のみを回転させた場合に、既存実装と同程度か、それ以上に破綻した occluder が出ないこと（極端な透け方が減ること）。
- 2D 図形との互換性:
  - 既存の正方形テスト（`tests/effects/test_culling_hidden_line.py`）が引き続き緑であること。
  - 厚み 0 / thickness 小のケースで、挙動が現行計画と大きく矛盾しないこと。

実装タスク（チェックリスト）

- 設計・準備
  - [ ] face 抽出の判定条件を決める（ループ判定・平面性しきい値 `plane_eps`・面積しきい値 `area_eps`・前向き/後向き判定）。
  - [ ] 3D 対応により 2D の挙動が変わり過ぎないよう、テストでカバーすべきパターンを列挙。
- 実装
  - [x] `src/effects/culling.py`: Geometry から閉ループ face を抽出するヘルパ（例: `_extract_faces_from_geometry(g: Geometry) -> list[np.ndarray]`）を追加。
  - [x] `src/effects/culling.py`: face 法線推定と前向き判定ロジックを実装。
  - [x] `_build_occluder_region` を拡張して face-aware パス → polygonize パス → buffer フォールバックの順で occluder を構築する。
- テスト
  - [x] 新規テスト（例: `tests/effects/test_culling_hidden_line_3d.py`）を追加し、立方体×2 の隠線が角度変更時にも大きく破綻しないことを確認。
  - [x] 既存の 2D テスト（`tests/effects/test_culling_hidden_line.py`）が引き続き通ることを確認。
- 品質・ドキュメント
  - [x] 変更ファイルに対する `ruff/black/isort/mypy` を実行。
  - [ ] `docs/plans/2025-11-15_effects_culling_hidden_line_v2.md` と本 3D 計画との役割の違いを簡単に追記（2.5D vs 3D 強化）。
  - [ ] `architecture.md` の Optional Dependencies/shapely セクションに「3D 対応 culling は face 抽出 + polygonize を用いる」旨を整理。

この計画が問題なければ、このチェックリストに沿って 3D Geometry 対応の culling エフェクト実装改善を進める。***
