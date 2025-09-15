# シェイプ一覧と作法

## 目的
- 最小のパラメータで「扱いやすい」ポリライン集合を返すプリミティブ群を提供する。
- どのシェイプも `Geometry`（(N,3) float32 + offsets）に正規化される。

## 代表的なシェイプ

- `polygon(n_sides)`: 単位円に内接する正多角形。
- `grid(subdivisions=(x,y))`: 1×1 の正方形グリッド。
- `sphere(subdivisions, sphere_type)`: 半径1の球（スタイル切替）。
- `torus(major_radius, minor_radius, ...)`: トーラス。
- `capsule(radius, height, ...)`: カプセル形状。
- `cylinder(radius, height, segments)`: 円柱。
- `cone(radius, height, segments)`: 円錐。
- `polyhedron(polygon_type)`: 正多面体。
- `lissajous(freq_x, freq_y, ...)`: 2D/3D リサージュ曲線。
- `text(text, font_size, ...)`: フォント輪郭から線分を生成。
- `asemic_glyph(...)`: 擬似文字。
- `attractor(attractor_type, ...)`: ストレンジアトラクタ。

実装は `shapes/*.py` を参照。各シェイプは「関数」として `@shapes.registry.shape()` で登録します。スタブは関数のシグネチャを解析して `G.<name>(...)` の引数を自動生成します。

## 開発ガイド
- 形状は `@shape` で関数登録する（継承不要）。
- 関数は副作用なし・決定的であること。
- 2D 入力は Z=0 を付与して (N,3) に正規化すること。
- キャッシュは `api.shape_factory.ShapeFactory`（`G`）に集約される。
