# Effects Cheat Sheet (関数エフェクト早見表)

最小の使い方とパラメータの意味を短くまとめた早見表です。値の単位/正規化の有無を明記しています。

- rotate(pivot=(x,y,z), angles_rad=(ax,ay,az))
  - 角度: 0..1 正規化（内部で 2π ラジアン化）
- 例: `(E.pipeline.rotate(rotate=(0.25,0,0)).build())(g)`

- translate(delta=(dx,dy,dz))
  - 距離: 物理単位（mm相当）
- 例: `(E.pipeline.translate(offset_x=10).build())(g)`

- scale(pivot=(x,y,z), scale=(sx,sy,sz))
  - 倍率: 実数倍率（1=等倍）
- 例: `(E.pipeline.scale(scale=(2,2,1)).build())(g)`

- displace(amplitude_mm, spatial_freq, t_sec)
  - intensity: 実数（変位量の係数）
  - frequency: 実数 or (fx,fy,fz)
- 例: `(E.pipeline.displace(intensity=0.3, frequency=0.8).build())(g)`

- fill(mode='lines'|'cross'|'dots', density, angle_rad)
  - density: 0..1 正規化（内部で線/ドット数に写像）
  - angle: ラジアン
- 例: `(E.pipeline.fill(pattern='cross', density=0.7).build())(g)`

- repeat(count, offset, angles_rad_step, scale, pivot)
  - n_duplicates: 0..1→0..MAX（整数）
  - rotate: 0..1 正規化角（各軸）
- 例: `(E.pipeline.repeat(n_duplicates=0.3, offset=(5,0,0)).build())(g)`

- extrude(direction, distance, scale, subdivisions, center_mode)
  - distance/scale/subdivisions: 0..1 正規化
  - center_mode: "origin"（既定）or "auto"（押し出し先ラインの重心基準）
- 例: `(E.pipeline.extrude(distance=0.5, scale=0.7, center_mode="auto").build())(g)`

- offset(distance, join='round', segments_per_circle)
  - distance: 0..1→0..25mm、resolution: 0..1→段数
- 例: `(E.pipeline.offset(distance=0.4, resolution=0.6).build())(g)`

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
  - rotate: 0..1 正規化角、scale: 倍率
- 例: `(E.pipeline.affine(rotate=(0,0,0.5), scale=(1.2,1.2,1)).build())(g)`

補足: ここにない詳細は各ファイルの docstring を参照してください。
