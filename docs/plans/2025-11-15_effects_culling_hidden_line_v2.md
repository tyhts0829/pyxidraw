# culling エフェクト実装改善計画（ポリゴナイズベースの隠線領域）

目的

- `sketch/251115.py` のように立方体（polyhedron）を重ねたとき、手前の立方体に完全に隠れる奥の立方体の線分を「すべて」非表示にする。
- 具体的には、手前オブジェクトのシルエット内に完全に入る奥オブジェクトの線分は一切描画しない（隠線消去に近い 2.5D 表現）。
- 既存の `culling` エフェクト API（`E.pipeline.culling(geos=[...])`）は維持しつつ、内部実装を改善する。

現状挙動と問題点

- 現行 `culling` は「手前レイヤーの MultiLineString を線幅相当で buffer → union」した領域を `occluder_region` とし、奥側線分に対して `difference(occluder_region)` を取る。
  - occluder は「線の周辺」のみを覆うため、手前オブジェクトの内部領域（面の中）は必ずしも完全には覆われない。
  - 結果として、「線同士が交差している部分」だけが主に culling され、それ以外の“完全に隠れているはずの線”が残りやすい。
- 立方体の例:
  - 手前の立方体の投影範囲内に完全に入っている奥の立方体の辺でも、線の buffer の外側にある部分が残ってしまう。
  - 想定する「隠線レンダリング」よりもワイヤーフレーム的な見え方に近い。

要求仕様（振る舞い）

- 手前レイヤーの「シルエット（投影面）」を隠線領域として扱い、その内部にある奥レイヤーの線分はすべて削除する。
  - 手前オブジェクトの前面に完全に隠れる奥オブジェクトは、輪郭を含めて一切描画されない。
  - 部分的に隠れる場合は、隠れていない部分のみが残る（線分の切断は許容）。
- Z の扱いは現行仕様を継承:
  - Z が小さいほど手前（`front_is_larger_z=False` が既定）。
  - `z_metric` は `"mean"|"min"|"max"` のいずれか（既定 `"mean"`）。
- 2.5D 前提:
  - すべての処理は XY 投影上で行い、Z はソートと「どのレイヤーがどのレイヤーを隠すか」の優先順位決定にのみ使用する。

改善方針（概略）

- 手前レイヤーの MultiLineString から「面領域」を推定し、その面を occluder として使う。
  - Shapely の `polygonize` を用い、線分集合からポリゴン（シルエット/面）を構成する。
  - 得られたポリゴン集合を union して「隠線領域」とし、必要に応じて線幅相当の buffer をかける。
- `culling` 内部では、以下 2 段階の occluder 構築戦略を採用する:
  1. polygonize ベース: MultiLineString → `polygonize` → Polygon/MultiPolygon → buffer（優先）
  2. フォールバック: polygonize で十分な面が得られない場合は、現行どおり `ml.buffer(thickness_mm * 0.5)` に戻す
- これにより:
  - 立方体や多角形など「閉じた輪郭」を持つジオメトリでは、内部面が隠線領域になり、奥側の線分がしっかり消える。
  - 開いた線（フリーハンド線など）は従来と同程度の振る舞いを維持する。

設計詳細案

### 1. occluder 領域構築の再設計

- 新ヘルパ関数（イメージ）:

  ```python
  def _build_occluder_region(
      ml: MultiLineString,
      *,
      thickness_mm: float,
  ) -> BaseGeometry:  # Polygon | MultiPolygon | GeometryCollection
      ...
  ```

- 手順:
  - `polygonize` を使って MultiLineString からポリゴンを生成:

    ```python
    from shapely.ops import polygonize, unary_union

    polys = list(polygonize(ml))
    if polys:
        poly_union = unary_union(polys)
    else:
        poly_union = None
    ```

  - `poly_union` が有効（空でない / 面積が十分大きい）場合:
    - `layer_area = poly_union.buffer(thickness_mm * 0.5)` を occluder として採用。
      - thickness は線幅と境界のマージンとして扱う。
  - `poly_union` が得られない場合（開いた線だけなど）:
    - 現行の `ml.buffer(thickness_mm * 0.5)` にフォールバック。
- 注意点:
  - 極端に小さいポリゴン（数値ノイズ）は area しきい値で除外。
  - polygonize のコストが高い場合に備え、MultiLineString の頂点数や線分数に応じてスキップする閾値を設定してもよい（改善の余地として保留）。

### 2. culling 本体ロジックへの組み込み

- 現状:
  - `occluder_region` は `ml.buffer(...)` の union。
- 変更後:
  - `occluder_region` の更新箇所を `_build_occluder_region` 呼び出しに置き換え:

    ```python
    layer_area = _build_occluder_region(ml, thickness_mm=thickness_mm)
    if occluder_region is None:
        occluder_region = layer_area
    else:
        occluder_region = occluder_region.union(layer_area)
    ```

- 差分処理は現行どおり `visible_ml = ml.difference(occluder_region)` を継続。
  - これにより、手前シルエット内部にある線分は面として切り取られ、完全に隠れている部分は生き残らない。

### 3.「完全に隠れる奥のオブジェクト」判定（オプション）

- 現行/改善後とも、`difference` ベースでは「部分的に見えている線」だけが残る。
- 追加アイデア（将来オプション）:
  - `visible_ml` が空になったレイヤーについては「そのレイヤーは完全に隠れている」とみなし、対応する `Geometry` 全体を空に差し替える処理を入れる。
  - ただし、「ほぼ完全に隠れているがごく僅かだけ見えている」ケースでは挙動が変わるため、現段階では導入しない。

### 4. API / パラメータ

- 公開 API は現行のまま:
  - `culling(geos, *, thickness_mm=0.3, z_metric="mean", front_is_larger_z=False)`
  - `E.pipeline.culling(geos=[...], thickness_mm=..., z_metric=..., front_is_larger_z=...)`
- 新しいモード切り替えパラメータ（例: `occluder_mode="polygonize|buffer"`）は追加しない方針。
  - 実装上は内部定数として扱い、必要になった場合にのみ GUI/パラメータとして露出する。

### 5. 互換性とフォールバック

- Shapely 利用前提は現行と同様（`pytest.importorskip("shapely")` にも整合）。
- polygonize による面構築がうまくいかないケース:
  - 開いた線: フォールバック buffer により現行と近い挙動を維持。
  - 数値的に壊れた線形状: 例外を飲み込んでフォールバック。
- 既存スケッチ:
  - 基本的には「より多く隠線される」方向の変化となる。
  - 完全に隠れてほしいのに残ってしまっていた線が消えるのが主な差分。

テスト計画（改善版）

- 既存テスト維持:
  - `tests/effects/test_culling_hidden_line.py` の基本ケース（2 レイヤー矩形 / 単一レイヤー）はそのまま通ること。
- 新規テスト案:
  - 立方体シナリオ（polyhedron）:
    - 前: Z=0 の立方体、奥: Z=1 の立方体（同程度のサイズ）を `G.polyhedron` で生成し、`E.pipeline.affine()` で整列。
    - `culling` 適用前後で、奥側の Geometry の線分数が大きく減少し、少なくとも「手前立方体の投影範囲内」にある線がほぼ消えていることを確認。
  - polygonize 成功/失敗ケース:
    - 閉じた矩形の MultiLineString → polygonize が多角形を返し、`occluder_region` が矩形面になること。
    - 単一開線（直線）の MultiLineString → polygonize が空となり、buffer ベースにフォールバックすること。
  - オーバーラップしないレイヤー:
    - 手前と奥の矩形が XY で交差しない場合、どちらの Geometry にも変化がない（線分数と座標が一致）こと。
  - z_metric の違い:
    - 同じ XY 位置に Z=0, Z=1 の 2 レイヤーを重ね、`z_metric="max"` / `"min"` / `"mean"` でソート順と culling 結果が変わらない（このケースでは等価）こと。

実装タスク（チェックリスト）

- 設計・準備
  - [x] polygonize ベース occluder の仕様を確定（面積しきい値・buffer 戦略・フォールバック条件）。
  - [ ] 既存ドキュメント（`docs/plans/2025-11-15_effects_culling_hidden_line.md`）との整合ポイントをメモ。
- 実装
  - [x] `src/effects/culling.py`: `_build_occluder_region(ml, thickness_mm)` ヘルパを追加。
  - [x] `culling(...)` 内で occluder_region 更新ロジックを `_build_occluder_region` に置き換え。
  - [x] polygonize が失敗した場合に現行 buffer ロジックへ安全にフォールバックする。
- テスト
  - [x] `tests/effects/test_culling_hidden_line.py`: polygonize ベースの挙動を検証する新ケースを追加（立方体 / 矩形）。
  - [x] Shapely 非導入環境でのスキップ条件が従来どおり機能することを確認。
- 品質・ドキュメント
  - [x] 変更ファイルに対する `ruff/black/isort/mypy` を実行。
  - [ ] `architecture.md` の Optional Dependencies (shapely) に「effects.culling は polygonize を用いる」旨を追記。
  - [ ] 必要であれば `docs/spec/culling.md`（新規）を作成し、本改善の仕様を反映。

この計画が問題なければ、このチェックリストに従って `culling` の実装改善（polygonize ベースの隠線領域構築）を進める。***
