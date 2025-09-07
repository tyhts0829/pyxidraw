# エフェクト一覧と作法

## 目的
- `Geometry -> Geometry` の純関数オペレータ群を統一インターフェイスで提供する。
- パラメータの妥当性を明示し、`spec`/スタブ/テストと整合を保つ。

## 代表的なエフェクト

- `affine(pivot, angles_rad, scale)`: スケール→回転→移動の一括変換。
- `rotate(pivot, angles_rad)`: Z 回りなどの回転。
- `translate(dx, dy, dz)`: 平行移動。
- `displace(amplitude_mm, spatial_freq, t_sec)`: Perlin 風変位を加える。
- `fill(mode, angle_rad, density)`: 線/クロス/ドットで塗りつぶし。
- `dash(dash_length, gap_length)`: 連続線を破線化。
- `explode(factor)`: 原点から外向きに頂点を移動。
- `extrude(direction, distance, scale, subdivisions, center_mode)`: 2D を厚み方向へ押し出し。
- `offset(distance|distance_mm, join, segments_per_circle)`: オフセット曲線を生成。
- `trim(start, end)`: パスを部分切り出し。
- `wobble(amplitude, frequency, phase)`: 時間で揺らぎを付与。

実装は `effects/*.py` を参照。関数先頭の docstring と `__param_meta__`（あれば）を `scripts/gen_g_stubs.py` が読み取り、スタブに短い引数説明を埋め込みます。

## パラメータメタ（任意）

各関数に `__param_meta__ = {"param": {"type": "number", "min": 0.0, "max": 1.0, "choices": [...]}}` のような辞書を添えると、
- `validate_spec()` が追加検証を行う
- スタブに `# meta:` や `# choices:` コメントが生成され、IDE の補助になる

## 開発ガイド
- 新規追加は `effects/` に関数で実装し、`@effects.registry.effect()` を付ける。
- 関数シグネチャは `def name(g: Geometry, *, ...) -> Geometry` とし、位置可変引数は使わない。
- `__param_meta__` を極力付与して、値域と choices を明示する。
