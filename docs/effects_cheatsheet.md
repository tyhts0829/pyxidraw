# Effects Cheat Sheet (関数エフェクト早見表)

最小の使い方とパラメータの意味を短くまとめた早見表です。値の単位/正規化の有無を明記しています。

- rotation(center=(x,y,z), rotate=(rx,ry,rz))
  - 角度: 0..1 正規化（内部で 2π ラジアン化）
  - 例: `(E.pipeline.rotation(rotate=(0.25,0,0)).build())(g)`

- translation(offset_x, offset_y, offset_z)
  - 距離: 物理単位（mm相当）
  - 例: `(E.pipeline.translation(offset_x=10).build())(g)`

- scaling(center=(x,y,z), scale=(sx,sy,sz))
  - 倍率: 実数倍率（1=等倍）
  - 例: `(E.pipeline.scaling(scale=(2,2,1)).build())(g)`

- noise(intensity, frequency, time)
  - intensity: 実数（変位量の係数）
  - frequency: 実数 or (fx,fy,fz)
  - 例: `(E.pipeline.noise(intensity=0.3, frequency=0.8).build())(g)`

- filling(pattern='lines'|'cross'|'dots', density, angle)
  - density: 0..1 正規化（内部で線/ドット数に写像）
  - angle: ラジアン
  - 例: `(E.pipeline.filling(pattern='cross', density=0.7).build())(g)`

- array(n_duplicates, offset, rotate, scale, center)
  - n_duplicates: 0..1→0..MAX（整数）
  - rotate: 0..1 正規化角（各軸）
  - 例: `(E.pipeline.array(n_duplicates=0.3, offset=(5,0,0)).build())(g)`

- extrude(direction, distance, scale, subdivisions)
  - distance/scale/subdivisions: 0..1 正規化
  - 例: `(E.pipeline.extrude(distance=0.5, scale=0.7).build())(g)`

- buffer(distance, join_style, resolution)
  - distance: 0..1→0..25mm、resolution: 0..1→段数
  - 例: `(E.pipeline.buffer(distance=0.4, resolution=0.6).build())(g)`

- dashify(dash_length, gap_length)
  - 長さ: 物理単位（mm相当）
  - 例: `(E.pipeline.dashify(dash_length=2, gap_length=1).build())(g)`

- wobble(amplitude, frequency, phase)
  - amplitude: 物理単位、frequency: 空間周波数 [cycles/unit]
  - 例: `(E.pipeline.wobble(amplitude=1.5, frequency=(0.2,0.1,0)).build())(g)`

- wave(amplitude, frequency, phase)
  - amplitude: 物理単位、frequency: 空間周波数
  - 例: `(E.pipeline.wave(amplitude=0.5, frequency=0.3).build())(g)`

- transform(center, scale, rotate)
  - rotate: 0..1 正規化角、scale: 倍率
  - 例: `(E.pipeline.transform(rotate=(0,0,0.5), scale=(1.2,1.2,1)).build())(g)`

補足: ここにない詳細は各ファイルの docstring を参照してください。
