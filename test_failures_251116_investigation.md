# テスト失敗調査メモ（2025-11-16）

対象テスト:

- `tests/effects/test_fill_per_shape_params.py::test_fill_per_shape_params_cycle_and_angle`
- `tests/effects/test_mirror3d_basic.py::test_mirror3d_equator_mirror_doubles_count_and_flips_z`
- `tests/shapes/test_text_font_search.py::test_config_dirs_prepend`
- `tests/test_effect_partition.py::test_donut_excludes_inner_hole`
- `tests/ui/parameters/test_dpg_mount_smoke.py::test_dpg_parameter_window_mounts_and_closes`

---

## 1. fill per-shape params（cycle + angle）

- テスト: `tests/effects/test_fill_per_shape_params.py::test_fill_per_shape_params_cycle_and_angle`
- 失敗内容: `counts[1] > counts[0] >= 1` で `assert 4 > 5`（中央バケットより左バケットの線本数が多い）。

### 関連コード

- 実装: `src/effects/fill.py:170` 以降
  - グローバル共平面経路の中で:
    - グルーピング: `build_evenodd_groups(v2d_all, offsets)`
    - パラメータ適用: `density_seq[gi % len(density_seq)]`, `angle_seq[gi % len(angle_seq)]`
    - スキャンスパン: `_scan_span_for_angle_xy(v2d_all[:, :2], ang_i)`
    - 間隔決定: `spacing_glob = _spacing_from_height(scan_h, float(d))`
    - 線生成: `_generate_line_fill_evenodd_multi(g_coords, g_offsets, d, ang_i, spacing_override=spacing_glob)`

### 観察と原因

- 入力は XY 平面上の 3 つの正方形（左・中央・右）で、`Geometry.from_lines([sq0, sq1, sq2])` の順に渡している。
- `build_evenodd_groups` の仕様上、このケースでは 3 つの外環がそれぞれ独立したグループ `[[0], [1], [2]]` になる想定。
  - そのため `density=[5.0, 15.0]` はグループごとに `[5, 15, 5]`、`angle_rad=[0, π/2]` は `[0, π/2, 0]` とサイクル適用される前提。
- 実際の実装では、共平面経路で「線間隔 spacing」を決めるときに、各グループごとの高さではなく **全頂点 `v2d_all` のスキャンスパン** を使っている。
  - `scan_h = _scan_span_for_angle_xy(v2d_all[:, :2], ang_i)` は全体の XY 分布から決定される。
  - その `scan_h` に対して `density` を渡し、`spacing_glob` を求めたあと、各グループに同じ `spacing_glob` を適用している。
- その結果:
  - 密度 `d` は「各グループごとの目標本数」ではなく、「グローバルなスキャンスパンに対する間隔決定の係数」として働く。
  - 離散化と端数処理の影響で、同じ大きさの図形に対しても `d=5` と `d=15` の本数差が期待どおりに現れない（今回のケースでは、中央の 15 本相当グループより左の 5 本相当グループの方が線本数が多くなっている）。
- まとめると、このテストは「`density` 配列が図形ごとの線本数の大小関係を直接決める」という旧来の直感に基づいたアサーションをしているが、
  実装は「全体スパン基準の spacing を共有する設計」に変わっており、**per-shape density の相対本数保証がなくなったこと**が失敗原因。

---

## 2. mirror3d equator ミラー（赤道反転）

- テスト: `tests/effects/test_mirror3d_basic.py::test_mirror3d_equator_mirror_doubles_count_and_flips_z`
- 失敗内容: `zs` に `±0.5` が含まれることを期待しているが、実際は `[-0.25, …, 0.25]` のみで `±0.5` が存在しない。

### 関連コード

- 実装: `src/effects/mirror3d.py`
  - 回転軸・半空間クリップ:
    - `_basis_perp_axis`, `_compute_azimuth_plane_normals`
    - `_clip_polyline_halfspace_3d`, `_clip_polyline_wedge`
  - azimuth モード本体: `mirror3d(..., mode='azimuth')`
    - くさび内クリップ → 2n 個の回転 + 境界反射
    - `mirror_equator=True` のとき、最後に赤道面 `eq_n = ax` で `_reflect_across_plane` を適用。

### 観察と原因

- テスト入力:
  - `p_src`: z = `+0.5`、XY は角度 15°。
  - `p_other`: z = `-0.25`、XY は角度 100°。
  - `mirror3d(..., n_azimuth=2, axis=(0,0,1), phi0_deg=0, mirror_equator=True)` を適用。
- 実際に同じ条件で実行すると:
  - 出力オフセット数は 8（`len(offsets)-1 == 8`）でテストどおり。
  - しかし `coords[:, 2]` は `[-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25]` となり、`±0.5` が一切含まれない。
- 原因解析:
  - azimuth モードでは最初に「くさび領域」で半空間クリップを行う。
    - `n_azimuth=2, phi0_deg=0, axis=(0,0,1)` の場合、くさびは XY 平面における **x<=0, y>=0** の象限相当になる。
    - `p_src`（15°）は x>0 なので **くさび外** → `_clip_polyline_halfspace_3d` で完全に削除される。
    - `p_other`（100°）は x<0, y>0 なので **くさび内** → `src_lines` に残る。
  - その後の回転・境界反射・赤道反転はすべて「くさび内から得られた `src_lines`」だけに適用されるため、
    - z = `-0.25` の系列と、その赤道反転 z = `+0.25` だけが複製される。
    - z = `+0.5` の点はそもそもソースとして採用されない。
- したがって、テストが期待している「z=+0.5 の点も equator ミラーで z=-0.5 に反転して現れる」という挙動は、
  実装の「くさび内クリップ後だけを複製する」設計と整合しておらず、**くさびクリップで `p_src` が除外されていること**が失敗原因。

---

## 3. TextRenderer のフォント検索順序（config search_dirs が先頭に来ない）

- テスト: `tests/shapes/test_text_font_search.py::test_config_dirs_prepend`
- 失敗内容: `TextRenderer.get_font_path_list()` の先頭要素が OS フォント (`/System/Library/Fonts/...`) になっており、
  テストで用意したダミーフォント (`tmp_path/fonts/DummyFont-Regular.ttf`) が先頭になっていない。

### 関連コード

- フォント列挙:
  - `src/shapes/text.py:TextRenderer.get_font_path_list`
    - 設定 `fonts.search_dirs` → `util.fonts.resolve_search_dirs`
    - OS フォントディレクトリ → `util.fonts.os_font_dirs`
    - これらを `dirs.extend(...)` で一つのリストに結合。
    - 実際のファイル列挙は `glob_font_files(dirs, EXTENSIONS)` に委譲。
  - `src/util/fonts.py:glob_font_files`
    - ディレクトリごとに `d.glob("**/*{ext}")` でファイルを集め、`seen: set[Path]` に追加。
    - 最終的に `return sorted(seen)` で **パス全体をソート**。

### 観察と原因

- テストでは `util.utils.load_config` をモンキーパッチし、`fonts.search_dirs` に一時ディレクトリを 1 つだけ返すようにしている。
  - 期待: `TextRenderer.get_font_path_list()` が「設定ディレクトリのファイルを OS フォントより前に並べる」こと。
- しかし実装では:
  - 設定由来ディレクトリと OS フォントディレクトリを単純に `dirs.extend(...)` で結合したあと、
  - `glob_font_files` 内で **集合 `set` に集めた上で `sorted(seen)` を返している**ため、元の `dirs` の順序は完全に失われている。
  - macOS 環境では `/System/Library/Fonts/...` の方が一時ディレクトリ `/private/var/.../DummyFont-Regular.ttf` より辞書順で前に来るため、OS フォントが先頭になる。
- よって、テストが要求している「設定ディレクトリのフォントが検索順の先頭に来る」という仕様と、
  実装の「全ファイルをパス順にグローバルソートする」という挙動が噛み合っておらず、
  **`glob_font_files` でディレクトリ順が崩れていること**が直接の失敗原因。

---

## 4. partition ドーナツ形状で穴が埋まる問題

- テスト: `tests/test_effect_partition.py::test_donut_excludes_inner_hole`
- 失敗内容: ドーナツ形状（外周 200, 内周 90 の多角形）に対して `partition` を適用した結果、
  生成された三角セルの重心 `c2d` が「内周ポリゴン `inner_ring` の内側」に入ってしまうケースがあり、`assert not _pnpoly(c2d, inner_ring)` が落ちる。

### 関連コード

- 実装: `src/effects/partition.py`
  - 共平面判定と XY 整列: `choose_coplanar_frame`
  - Shapely 有効時のパス:
    - リング収集: `rings_xy`（`_ensure_closed` 済みの各リング）
    - 偶奇 XOR 合成: `region = poly if region is None else region.symmetric_difference(poly)`
      - outer + inner でドーナツ領域（穴付き Polygon）を構成する想定。
    - サイトサンプリング: `region.contains(_SPoint(rx, ry))` でドーナツ領域に制限。
    - Voronoi: `_voronoi_diagram(mp, envelope=region.envelope, edges=False)`
    - セル交差: `inter = cell.intersection(region)`
    - 多角形外周抽出: `_collect_polygon_exteriors(inter)`
    - 3D 復元: `transform_back(..., R_all, z_all)`
  - ホール処理: `_collect_polygon_exteriors`
    - `Polygon` の場合: `geom.exterior.coords` のみを使用し、**holes（interiors）は無視**する。

### 観察と原因

- Shapely 2.0.6 がインストールされており、このテストでは Shapely パスが使用される。
- `region` 自体は outer/inner の XOR により「穴付きポリゴン（ドーナツ領域）」として構成される想定だが、
  Voronoi セルとの交差 `cell.intersection(region)` の結果もまた、**穴付き Polygon や MultiPolygon** になり得る。
- 現状の `_collect_polygon_exteriors` は、`geom` が Polygon のとき:
  - `geom.exterior.coords` だけを ndarray に変換し、
  - `geom.interiors`（穴の輪郭）は完全に捨てている。
- そのため:
  - `region` の穴にまたがる Voronoi セルに対しては、「穴をくり抜いたポリゴン」ではなく、「穴を無視した外周だけのポリゴン」として出力される。
  - 結果として、理論上は空であるべきドーナツ穴の領域にもポリゴン（ループ）が残り、その一部の三角セル重心が `inner_ring` の内側に入ってしまう。
- まとめると、`partition` の Shapely 経路では **交差結果の holes を破棄しているため、ドーナツ穴が正しく除外されていない**ことが、テスト失敗の直接原因。

---

## 5. Dear PyGui パラメータウィンドウのマウント／クローズ

- テスト: `tests/ui/parameters/test_dpg_mount_smoke.py::test_dpg_parameter_window_mounts_and_closes`
- 失敗内容（ユーザー報告）:
  - 全体テストでは `AttributeError: module 'dearpygui.dearpygui' has no attribute 'hide_viewport'`。
  - 個別実行時はテストが長時間返ってこない印象（ユーザー側で中断）。

### 関連コード

- テスト側:
  - `pytest.importorskip("dearpygui.dearpygui")` により、Dear PyGui 未インストール環境ではテスト自体をスキップ。
  - `ParameterWindow(store=..., layout=..., auto_show=False)` を生成し、`set_visible(True/False)` と `close()` を呼ぶだけの smoke テスト。
- 実装側: `src/engine/ui/parameters/dpg_window.py`
  - Dear PyGui 直接依存: `import dearpygui.dearpygui as dpg`
  - コンストラクタ:
    - `dpg.create_context()`, `dpg.create_viewport(...)`, `dpg.setup_dearpygui()`
    - `auto_show=True` のとき `dpg.show_viewport()` と内部ドライバ起動。
  - 可視切り替え: `set_visible`
    - `visible=True`: `dpg.show_viewport()` → ドライバ起動。
    - `visible=False`: `dpg.hide_viewport()` → ドライバ停止。
  - 終了処理: `close`
    - 購読解除 → ドライバ停止 → `dpg.hide_viewport()` → `dpg.destroy_context()`。

### 観察と原因

- 現在の環境で Dear PyGui を調べると:

  ```python
  import dearpygui.dearpygui as dpg
  hasattr(dpg, "hide_viewport")  # False
  hasattr(dpg, "show_viewport")  # True
  ```

- つまり、この Dear PyGui バージョンには `hide_viewport` API が存在しない。
  - 一方で実装は `set_visible(False)` や `close()` の中で **無条件に `dpg.hide_viewport()` を呼んでいる**。
- そのため:
  - テストが `win.set_visible(False)` もしくは `win.close()` を通過したタイミングで `AttributeError` が発生し、テストが失敗する。
  - 個別実行時にテストが「止まって見える」のは、Dear PyGui の初期化や内部ドライバスレッドが GUI イベントループに依存していることに加え、
    例外が出るまでに時間がかかっている可能性が高い（ただし、根本原因はあくまで `hide_viewport` の欠如）。
- 結論として、このテストの失敗原因は **Dear PyGui のバージョンと実装が想定する API セットの不整合（`hide_viewport` が存在しない）** によるもの。

