# エフェクト引数一覧

`src/effects/` 配下の全エフェクト関数の引数名と型注釈の一覧。

## affine
- モジュール: `src/effects/affine.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `auto_center`: `bool`
  - `pivot`: `Vec3`
  - `angles_rad`: `Vec3`
  - `scale`: `Vec3`
  - `delta`: `Vec3`

## boldify
- モジュール: `src/effects/boldify.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `boldness`: `float`

## clip
- モジュール: `src/effects/clip.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `outline`: `Geometry | Sequence[Geometry]`
  - `draw_outline`: `bool`
  - `draw_inside`: `bool`

## collapse
- モジュール: `src/effects/collapse.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `intensity`: `float`
  - `subdivisions`: `int`

## culling
- モジュール: `src/effects/culling.py`
- 戻り値: `list[Geometry]`
- 引数:
  - `geos`: `Sequence[Any]`
  - `thickness_mm`: `float`
  - `z_metric`: `str`
  - `front_is_larger_z`: `bool`

## dash
- モジュール: `src/effects/dash.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `dash_length`: `float | list[float] | tuple[float, ...]`
  - `gap_length`: `float | list[float] | tuple[float, ...]`
  - `offset`: `float | list[float] | tuple[float, ...]`

## displace
- モジュール: `src/effects/displace.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `amplitude_mm`: `float | Vec3`
  - `spatial_freq`: `float | Vec3`
  - `t_sec`: `float`

## explode
- モジュール: `src/effects/explode.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `factor`: `float`

## extrude
- モジュール: `src/effects/extrude.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `direction`: `Vec3`
  - `distance`: `float`
  - `scale`: `float`
  - `subdivisions`: `int`
  - `center_mode`: `Literal['origin', 'auto']`

## fill
- モジュール: `src/effects/fill.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `angle_sets`: `int | list[int] | tuple[int, ...]`
  - `angle_rad`: `float | list[float] | tuple[float, ...]`
  - `density`: `float | list[float] | tuple[float, ...]`
  - `remove_boundary`: `bool`

## mirror
- モジュール: `src/effects/mirror.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `n_mirror`: `int`
  - `cx`: `float`
  - `cy`: `float`
  - `source_side`: `bool | Sequence[bool]`
  - `show_planes`: `bool`

## mirror3d
- モジュール: `src/effects/mirror3d.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `n_azimuth`: `int`
  - `cx`: `float`
  - `cy`: `float`
  - `cz`: `float`
  - `axis`: `Sequence[float]`
  - `phi0_deg`: `float`
  - `mirror_equator`: `bool`
  - `source_side`: `bool | Sequence[bool]`
  - `mode`: `str`
  - `group`: `str | None`
  - `use_reflection`: `bool`
  - `show_planes`: `bool`

## offset
- モジュール: `src/effects/offset.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `join`: `str`
  - `segments_per_circle`: `int`
  - `distance`: `float`

## partition
- モジュール: `src/effects/partition.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `site_count`: `int`
  - `seed`: `int`

## repeat
- モジュール: `src/effects/repeat.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `count`: `int`
  - `offset`: `Vec3`
  - `angles_rad_step`: `Vec3`
  - `scale`: `Vec3`
  - `auto_center`: `bool`
  - `pivot`: `Vec3`

## rotate
- モジュール: `src/effects/rotate.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `auto_center`: `bool`
  - `pivot`: `Vec3`
  - `angles_rad`: `Vec3`

## scale
- モジュール: `src/effects/scale.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `auto_center`: `bool`
  - `pivot`: `Vec3`
  - `scale`: `Vec3`

## style
- モジュール: `src/effects/style.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `color`: `tuple[float, float, float] | Iterable[float] | None`
  - `thickness`: `float`

## subdivide
- モジュール: `src/effects/subdivide.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `subdivisions`: `int`

## translate
- モジュール: `src/effects/translate.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `delta`: `Vec3`

## trim
- モジュール: `src/effects/trim.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `start_param`: `float`
  - `end_param`: `float`

## drop
- モジュール: `src/effects/drop.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `interval`: `int | None`
  - `offset`: `int`
  - `min_length`: `float | None`
  - `max_length`: `float | None`
  - `probability`: `float`
  - `by`: `Literal['line', 'face']`
  - `seed`: `int | None`
  - `keep_mode`: `Literal['keep', 'drop']`

## twist
- モジュール: `src/effects/twist.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `angle_rad`: `float`
  - `axis`: `str`

## weave
- モジュール: `src/effects/weave.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `num_candidate_lines`: `int`
  - `relaxation_iterations`: `int`
  - `step`: `float`

## wobble
- モジュール: `src/effects/wobble.py`
- 戻り値: `Geometry`
- 引数:
  - `g`: `Geometry`
  - `amplitude`: `float`
  - `frequency`: `float | Vec3`
  - `phase`: `float`
