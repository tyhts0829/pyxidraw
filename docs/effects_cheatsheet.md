# Effects Cheat Sheet (関数エフェクト早見表)

最小の使い方とパラメータの意味を短くまとめた早見表です。値の単位/正規化の有無を明記しています。

- rotate(pivot=(x,y,z), angles_rad=(ax,ay,az))
  - 角度: ラジアン（推奨）。互換として `rotate=(0..1)` も受理。
- 例: `(E.pipeline.rotate(angles_rad=(0.5*3.14159,0,0)).build())(g)`

- translate(delta=(dx,dy,dz))
  - 距離: 物理単位（mm相当）
- 例: `(E.pipeline.translate(delta=(10,0,0)).build())(g)`

- scale(pivot=(x,y,z), scale=(sx,sy,sz))
  - 倍率: 実数倍率（1=等倍）
- 例: `(E.pipeline.scale(scale=(2,2,1)).build())(g)`

- displace(amplitude_mm, spatial_freq, t_sec)
  - amplitude_mm: 実数（変位量の係数）
  - spatial_freq: 実数 or (fx,fy,fz)
- 例: `(E.pipeline.displace(amplitude_mm=0.3, spatial_freq=0.8).build())(g)`

- fill(mode='lines'|'cross'|'dots', density, angle_rad)
  - density: 0..1 正規化（内部で線/ドット数に写像）
  - angle_rad: ラジアン
- 例: `(E.pipeline.fill(mode='cross', density=0.7).build())(g)`

- repeat(count, offset, angles_rad_step, scale, pivot)
  - count: 複製数（整数）
  - angles_rad_step: ラジアン
- 例: `(E.pipeline.repeat(count=3, offset=(5,0,0)).build())(g)`

- extrude(direction, distance, scale, subdivisions, center_mode)
  - distance/scale/subdivisions: 0..1 正規化
  - center_mode: "origin"（既定）or "auto"（押し出し先ラインの重心基準）
- 例: `(E.pipeline.extrude(distance=0.5, scale=0.7, center_mode="auto").build())(g)`

- offset(distance, join='round', segments_per_circle)
  - distance: 0..1→0..25mm
  - segments_per_circle: 円の分割数（整数）
- 例: `(E.pipeline.offset(distance=0.4, segments_per_circle=12).build())(g)`

- dash(dash_length, gap_length)
  - 長さ: 物理単位（mm相当）
- 例: `(E.pipeline.dash(dash_length=2, gap_length=1).build())(g)`

- wobble(amplitude, frequency, phase)
  - amplitude: 物理単位、frequency: 空間周波数 [cycles/unit]
- 例: `(E.pipeline.wobble(amplitude=1.5, frequency=(0.2,0.1,0)).build())(g)`

- ripple(amplitude, frequency, phase)
  - amplitude: 物理単位、frequency: 空間周波数
- 例: `(E.pipeline.ripple(amplitude=0.5, frequency=0.3).build())(g)`

- affine(pivot, scale, angles_rad)
  - angles_rad: ラジアン、scale: 倍率
- 例: `(E.pipeline.affine(angles_rad=(0,0,1.5708), scale=(1.2,1.2,1)).build())(g)`

補足: ここにない詳細は各ファイルの docstring を参照してください。

## 命名と登録ポリシー（更新）

- レジストリキーの正規化は統一ポリシー（ADR 0012）に従います。
  - `-` → `_` に置換、CamelCase → snake_case、すべて小文字。
  - エイリアス登録はサポートしません。明示名で登録してください。

## パラメータメタデータ（validate_spec 用）

- エフェクトは任意で `__param_meta__` を宣言できます。
  - `type`: `number` | `integer` | `string` | `vec3`（scalar/1-tuple/3-tuple を許容）
  - `min`/`max`: 数値域制約（任意）
  - `choices`: 許容値の列挙（任意）

例（抜粋）:

```python
# effects/translation.py
translate.__param_meta__ = {
    "delta": {"type": "vec3"},
}

# effects/rotation.py
rotate.__param_meta__ = {
    "pivot": {"type": "vec3"},
    "angles_rad": {"type": "vec3"},
}

# effects/scaling.py
scale.__param_meta__ = {
    "pivot": {"type": "vec3"},
    "scale": {"type": "vec3"},
}
```

## 主要エフェクト パラメータ一覧（表）

| エフェクト | パラメータ | 型 | 範囲/choices | 単位/備考 |
|---|---|---|---|---|
| translate | delta | vec3 | — | 平行移動量（mm 相当） |
| rotate | pivot | vec3 | — | 回転中心 |
| rotate | angles_rad | vec3 | — | 各軸ラジアン角 |
| scale | pivot | vec3 | — | スケール中心 |
| scale | scale | vec3 | — | 倍率（1=等倍） |
| displace | amplitude_mm | number | min 0 | 変位の大きさ（mm 相当） |
| displace | spatial_freq | number or vec3 | — | 空間周波数 [cycles/unit] |
| displace | t_sec | number | min 0 | 時間オフセット（秒） |
| fill | mode | string | lines/cross/dots | パターン種別 |
| fill | density | number | 0..1 | 0..1 正規化（線/ドット数に写像） |
| fill | angle_rad | number | — | ラジアン角 |
| repeat | count | integer | ≥0 | 複製数 |
| repeat | offset | vec3 | — | 複製ごとの移動量 |
| repeat | angles_rad_step | vec3 | — | 複製ごとの回転増分（ラジアン） |
| repeat | scale | vec3 | — | 複製ごとのスケール倍率 |
| repeat | pivot | vec3 | — | 変換の基準点 |
| extrude | direction | vec3 | — | 押し出し方向ベクトル（正規化不要） |
| extrude | distance | number | 0..1 | 押し出し距離係数（内部レンジへ写像） |
| extrude | scale | number | 0..1 | 押し出し側スケール係数 |
| extrude | subdivisions | number | 0..1 | 細分化係数（整数へ写像） |
| extrude | center_mode | string | origin/auto | auto は押し出し先重心基準 |
| offset | distance | number | 0..1 | 0..1→mm へ写像 |
| offset | segments_per_circle | integer | ≥4 推奨 | 円弧近似の分割数 |
| dash | dash_length | number | >0 | 実長（mm 相当） |
| dash | gap_length | number | ≥0 | 実長（mm 相当） |
| wobble | amplitude | number | ≥0 | 実長（mm 相当） |
| wobble | frequency | number or vec3 | — | 空間周波数 |
| wobble | phase | number | — | ラジアン位相 |
| ripple | amplitude | number | ≥0 | 実長（mm 相当） |
| ripple | frequency | number or vec3 | — | 空間周波数 |
| ripple | phase | number | — | ラジアン位相 |
| twist | angle | number | — | 最大ねじれ角（度） |
| twist | axis | string | x/y/z | ねじれ軸 |

注:
- 0..1 正規化系のパラメータは内部で上限レンジへ写像されます（詳細は各実装参照）。
- vec3 は「スカラー/1要素/3要素」を許容（Spec 検証でサポート済み）。
