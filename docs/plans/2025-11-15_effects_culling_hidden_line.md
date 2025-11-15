# culling エフェクト実装計画（Z ソート＋ Shapely 隠線クリップ）

目的

- ユーザーは `sketch/251115.py` のように `E.pipeline.culling(geos=[p1(g1), p2(g2), p3(g3)])` と書くだけで、複数 Geometry 間の隠線処理結果を得られるようにする（この UX を前提とした設計とする）。
- `Geometry | LazyGeometry` の Sequence を Z 座標に基づいて奥 → 手前の関係で解釈し、手前オブジェクトに隠れる線分を奥側からクリップする単一エフェクトを提供する。
- Shapely による 2.5D 的な隠線処理で、ペンプロッタ描画でも奥行き表現をしやすくする。

背景（現状の問題）

- 現状、`user_draw(t)` が `Sequence[Geometry | LazyGeometry]` を返した場合でも、「手前のオブジェクトに隠れる線」を自動で落とす仕組みが無い。
- 近い要素から遠い要素へ重ね書きすることで「なんとなく」奥行き表現は可能だが、完全に隠れる線分まで描かれてしまい、視覚的にごちゃつく。
- `effects.clip` には Shapely ベースのクリップ処理が存在するが、単一 `Geometry` + マスク（outline）向けであり、Z ソートされた複数オブジェクト間の隠線処理はカバーしていない。

スコープ / 非目標

- スコープ
  - 入力: `Sequence[Geometry | LazyGeometry]`（= レイヤー列の想定）。
    - ユーザー視点では `E.pipeline.culling(geos=[p1(g1), p2(g2), p3(g3)])` の `geos` 引数として渡す Geometry 列をこの Sequence とみなす。
  - 各要素は「ほぼ一定の Z（平行平面上）」に乗っていると仮定し、その代表値（例: 平均 Z）で奥行きを決める。
  - Z 軸は「値が大きいほど奥」という前提で扱い、デフォルトでは Z が小さいレイヤーほど手前にあると見なす。
  - 画面は XY 平面、Z はビュー方向の奥行き指標としてのみ使用する（投影自体は既に XY に済んでいる前提）。
  - Shapely を前提依存とし、線分の buffer（太さヒント）を用いた「面としての占有領域」に対する差分で隠線を落とす。
- 非目標
  - 一般の 3D 可視化（任意の姿勢の多面体/メッシュに対する厳密な隠線/隠面消去）。
  - 透過・半透明・マテリアルなどのフォトリアルなレンダリング。
  - Renderer/Runtime 側の描画順制御や Z バッファ互換機能の追加。

インターフェース案

- ユーザー向け API

  ```python
  from api import E, G


  def draw(t: float):
      g1 = G.polyhedron().scale(40, 40, 40)
      p1 = E.pipeline.affine()
      g2 = G.polyhedron().scale(30, 30, 30)
      p2 = E.pipeline.affine()
      g3 = G.polyhedron().scale(20, 20, 20)
      p3 = E.pipeline.affine()
      # geos には Geometry | LazyGeometry の列を渡す
      geos = [p1(g1), p2(g2), p3(g3)]
      return E.pipeline.culling(
          geos=geos,
          thickness_mm=0.3,
          z_metric="mean",
          front_is_larger_z=False,
      )
  ```

- 内部実装モジュール: `src/effects/culling.py`（通常のエフェクトモジュール）。
- コア関数（Sequence を受け取る純関数エフェクト）

  ```python
  from collections.abc import Sequence
  from engine.core.geometry import Geometry
  from engine.core.lazy_geometry import LazyGeometry

  from .registry import effect


  @effect()
  def culling(
      geos: Sequence[Geometry | LazyGeometry],
      *,
      thickness_mm: float = 0.3,
      z_metric: str = "mean",           # "mean" | "min" | "max"
      front_is_larger_z: bool = False,  # False: 大きい Z が奥（既定）
  ) -> list[Geometry]:
      ...
  ```

- 振る舞い
  - `geos` を Z 指標（`z_metric`）でスカラー化し、`front_is_larger_z` に応じて手前 → 奥順にソートする。
    - 既定値 `front_is_larger_z=False` の場合は「Z が小さいほど手前」として扱う。
  - `geos` 内の `Geometry` / `LazyGeometry` をそのまま受け取り、必要最小限の実体化で処理する（Lazy を最大限活かす）。
  - 手前レイヤーの「占有領域」（buffer で膨らませた MultiLineString）を union しつつ奥側の線分に対して `difference` を取り、隠れた部分を削除する。
  - 出力は元の入力順（インデックス順）で並べ替えて返す（Z ソートは内部処理のみに使用）。

アルゴリズム設計

### 1. 入力正規化（Lazy を活かす）

- シグネチャで `Sequence[Geometry | LazyGeometry]`（ユーザー引数名 `geos`）を受け取り、内部では以下の構造に正規化する。

  ```python
  @dataclass
  class _Layer:
      src: Geometry | LazyGeometry  # 元オブジェクト（遅延を保持）
      z_value: float                # 深度指標
      index: int                    # 入力順
  ```

- 処理
  - `geos` を列挙し、`Geometry` はそのまま、`LazyGeometry` は「Z 推定のためにだけ」軽量実体化する方針を採用する。
    - 第 1 フェーズでは単純に `LazyGeometry.realize()` を呼び、以後の処理でそのキャッシュを再利用する。
    - `LazyGeometry.realize()` 自体が shape/prefix キャッシュを持つため、フレーム間で同一 `LazyGeometry` を再利用する限り、重い計算は 1 度だけで済む想定。
    - 後続の最適化として「Z 代表値だけを推定する専用ヘルパ」を `LazyGeometry` 側に追加する余地がある（本計画では非スコープ）。
  - 空ジオメトリ（`n_vertices == 0`）はそのままスキップ or 通過（スキップしても結果には影響しない）。
  - `z_metric` に応じて `coords[:, 2]` から `mean/min/max` を計算し `z_value` に格納。
  - 入力がすべて Z=0 の場合でも、Z の差が無いだけでアルゴリズムはそのまま動作（この場合は元の描画順が優先される）。

### 2. 深度ソート

- `_Layer` のリストを `key=lambda L: L.z_value` でソート。
  - `front_is_larger_z=False`（既定）の場合: `sorted(..., key=...)` → Z 小さい順（手前 → 奥）。
  - `True` の場合は降順ソート。
- 以降の隠線処理は「手前から奥へ」順に走査する。

### 3. Shapely 変換（線分 → MultiLineString）

- 各 `_Layer.src` について、必要時に `Geometry` へ実体化し（`LazyGeometry` なら `realize()`）、Shapely の `MultiLineString` 表現に変換するヘルパを用意。

  ```python
  def _geometry_to_multiline(g: Geometry) -> "MultiLineString":
      coords, offsets = g.as_arrays(copy=False)
      lines = []
      for i in range(len(offsets) - 1):
          seg = coords[offsets[i] : offsets[i + 1]]
          if seg.shape[0] >= 2:
              lines.append(seg[:, :2])  # XY だけ渡す
      if not lines:
          return MultiLineString()
      return MultiLineString(lines)
  ```

- import 方針
  - `from shapely.geometry import MultiLineString, Polygon` を処理関数内でローカル import する（`architecture.md` の Optional Dependencies 方針に従う）。
  - shapely 未導入環境では ImportError をそのまま上げる（`tools/dummy_deps.py` により開発環境ではシムが入る想定）。

### 4. 手前マスク（占有領域）の構築

- 手前から奥へ走査しながら、次の 2 つを進行形で更新する。
  - `occluder_region: Polygon | MultiPolygon | None`（手前オブジェクトの union buffer）
  - `visible_parts: dict[int, Geometry]`（index → 可視部分 Geometry）
- 各 `_Layer` について:

  1. 必要時に `Geometry` を取得
     - `geom = layer.src if isinstance(layer.src, Geometry) else layer.src.realize()`
     - `LazyGeometry` からの `realize()` は内部キャッシュにより 1 回のみ実行される前提。
  2. `ml = _geometry_to_multiline(geom)`
  3. `occluder_region` が `None` なら:
     - `visible_ml = ml`（最前面は隠れることがない）
  4. `occluder_region` がある場合:
     - `visible_ml = ml.difference(occluder_region)`（Polygon による線分の差分）
  5. `visible_ml` を `Geometry` に戻して `visible_parts[layer.index]` に保存。
  6. レイヤー自身の占有領域を更新:
     - `layer_area = ml.buffer(thickness_mm * 0.5, join_style=..., cap_style=...)`
     - `occluder_region = layer_area` if `None` else `occluder_region.union(layer_area)`

- `thickness_mm` は「線が占める厚みの近似値」として使い、視覚上十分に隠れているとみなす幅を指定する。
  - デフォルトは 0.3mm 程度を想定（後から調整可能）。
  - 必要に応じて join/cap スタイルは `round` を既定値とする（滑らかさ優先）。

### 5. Shapely 結果の Geometry への再変換

- `effects.clip` に既存の `_to_lines_from_shapely` があるため、同等のヘルパを `culling` 側にも用意するか、共通化する。
  - `MultiLineString` / `LineString` / `GeometryCollection` から `list[np.ndarray]` の XY 配列列を得る。
  - 各配列に対し、レイヤーの `z_value` を Z として付与し `(N, 3)` にする。
- `Geometry.from_lines` で `Geometry` にまとめる。
- レイヤーが完全に隠れている場合は空 `Geometry` を返す（`coords.shape==(0,3), offsets==[0]`）。

### 6. 戻り値の並べ替えと整形

- `visible_parts`（index→Geometry）を元の入力順に並べ直し、`list[Geometry]` として返す。
- `LazyGeometry` に戻す必要は無い想定（隠線処理は「最終段」の加工であり、ここからさらにエフェクトパイプラインを積むケースは稀とみなす）。

テスト計画

- 新規テストファイル: `tests/effects/test_culling_hidden_line.py`（仮）
- 想定ケース
  - 単純な 2 レイヤー矩形
    - 手前: Z=0, 奥: Z=1 の二つの矩形（同一 XY 領域）を用意し、奥側が完全に隠れることを確認。
    - 交差するだけ（完全包含ではない）場合に、交差部分のみが奥側から削られることを確認。
  - 3 レイヤー（階段状）
    - Z=0,1,2 の順に少しずつオフセットした矩形を用意し、可視線が期待通り階段状になるかを確認。
  - thickness の影響
    - 非常に小さい thickness ではほぼノーカット、大きい thickness ではかなり手前の領域まで削られることを確認。
  - LazyGeometry 入力
    - `G.cube(...).rotate(...).translate(...).pipe(...)` のような LazyGeometry を複数作り、Sequence で渡したときに例外なく実行されること。
  - sketch/UX パターン
    - `sketch/251115.py` と同形式で `E.pipeline.affine()` から 3 つの Geometry を作り、`E.pipeline.culling(geos=[...])` が例外なく動作し、期待通りの隠線結果を返すこと。
  - エッジケース
    - 空の Sequence（`[]`）で空リストを返す。
    - 単一要素のみの Sequence では入力と同じ形状が返る。

ドキュメント / 設計同期

- 実装後に更新すべき候補
  - `architecture.md`
    - Optional Dependencies の shapely 欄に `effects.culling` を追記。
    - 「パイプライン（Geometry→Renderer）」周辺に「Sequence レベルの隠線処理ユーティリティ」として簡単に触れる（必要なら）。
  - `docs/spec/` 配下
    - `docs/spec/culling.md`（新規）として、API シグネチャと制約（2.5D / thickness ベース）をまとめる案。
  - `README.md`
    - 使用例として 1 つだけ簡単なコードスニペットを追加するかは任意。

実装タスク（チェックリスト / TDD プロセス込み）

- 仕様・設計
  - [x] コア仕様の確定（thickness 既定値・LazyGeometry の扱いの細部）
  - [x] テストケースの具体化（テスト計画セクションから入力/期待結果を明文化）
- TDD サイクル（1 スライスごとに繰り返し）
  - [x] 失敗するテストを追加（`tests/effects/test_culling_hidden_line.py`）
  - [x] 最小限の実装を追加/変更してテストを緑にする
  - [x] 実装とテストのリファクタリング（重複削減・分割）＋テスト再実行で緑を確認
- 実装
  - [x] `src/effects/culling.py`: モジュール追加 + ヘッダ docstring
  - [x] `src/effects/culling.py`: `_geometry_to_multiline` / Shapely→ndarray 変換ヘルパ実装
  - [x] `src/effects/culling.py`: `culling(...)` 本体実装
  - [ ] `src/effects/culling.py`: 必要に応じて簡易結果キャッシュ（layers 署名 + thickness など）を追加
- ドキュメント / 品質
  - [ ] `architecture.md` / `docs/spec` の関連箇所更新
  - [x] 変更ファイルに対する `ruff/black/isort/mypy` 実行
  - [x] テスト: `pytest -q tests/effects/test_culling_hidden_line.py`

リスクと対応

- 性能
  - Shapely の `buffer` + `union` はレイヤー数・線分数が多いと重くなる可能性がある。
  - 対応方針: 初期実装ではシンプルな実装を優先し、ボトルネックになった場合に AABB ベースの早期スキップやタイル分割（clip の `_prepare_projection_mask` 相当）を検討する。
- Z の定義
  - 本エフェクトでは「Z が大きいほど奥」という前提を採用し、既定では Z が小さい方を手前として扱う。
  - 必要に応じて `front_is_larger_z=True` で向きを反転できるようにする。
- 形状の前提
  - 現実の 3D モデルのような完全な隠線消去ではなく、「線の太さで膨らませた占有領域に隠れる部分を削る」近似である。
  - ドキュメントで 2.5D 的な近似であることを明記する。

質問 / 確認事項

- [x] Z の向き: 「Z が大きいほど奥」という前提を採用（既定は `front_is_larger_z=False`）。
- [ ] thickness の既定値: 0.3mm 程度で良いか、それとももう少し大きめ/小さめが良いか。
- [x] 出力型: `list[Geometry]` のみに正規化してよいか（Lazy に戻さない設計で問題ないか）。
- [x] API 露出: ユーザーは `E.pipeline.culling(geos=[...])` を主入口として利用し、`effects.culling.culling` は内部実装用の関数とする。

以上、特に問題なければこの計画に沿って実装を進めます（上記チェックリストを順次埋めていきます）。
