# palette TRI/TET プレビュー消失バグ修正計画（ドラフト）

目的: Parameter GUI の Palette セクションで palette type を TRI/TET に切り替えた際、プレビューのカラーボタンが一切表示されなくなる問題を解消し、常に有効な Palette オブジェクトとスウォッチを生成できるようにする。

## 現象と再現イメージ

- Parameter GUI を有効にしたスケッチで起動。
- Palette セクションにて:
  - `palette.type` を `TRI` または `TET` に変更。
  - `palette.n_colors` はデフォルト（4）や任意の値（2〜6）に設定されている想定。
- 結果:
  - Preview 行のカラースウォッチが 0 個になり、何も表示されない。
  - `palette.type` を ANA/COM/SPL/TAS などに戻すとスウォッチが再び表示される。

## 原因（実装レベルの整理）

- プレビュー構築経路
  - `src/engine/ui/parameters/dpg_window_content.py:1231`  
    - `_refresh_palette_preview()` が ParameterStore から値を読み出し、パレットを再計算する。
    - `engine.ui.palette.helpers.build_palette_from_values(...)` を呼び出して `palette.Palette` を生成。
    - 生成失敗時は `palette_obj = None` として握りつぶし、`palette_obj is None` の場合はそのまま `return` してスウォッチを構築しない。
  - `palette_obj` が得られた場合のみ、`palette.ui_helpers.export_palette(..., ExportFormat.HEX / SRGB_01)` で色リストを取り出し、`dpg.add_color_button(...)` でスウォッチを追加している。
- Palette 生成ヘルパ
  - `src/engine/ui/palette/helpers.py`
    - `build_palette_from_values(...)` は `palette_type_value` / `palette_style_value` / `n_colors_value` などの UI 値を正規化した後、`palette.generate_palette(...)` を呼び出す。
    - `palette.type` の値は `"ANA" / "COM" / "SPL" / "TRI" / "TET" / "TAS"` であり、`_TYPE_BY_SHORT_LABEL` 経由で `PaletteType.TRIADIC` / `TETRADIC` に正しく解決される。
    - `n_colors_value` は int 化後、そのまま `n_colors` として `generate_palette(...)` に渡される。
- Palette コア側の制約
  - `src/palette/harmony.py`
    - `compute_hue_offsets(palette_type, n_colors)` 内で、`PaletteType.TRIADIC` と `PaletteType.TETRADIC` に対して次の制約がある:
      - TRIADIC: `if n_colors != 3: raise ValueError("Triadic palette is defined for n_colors=3.")`
      - TETRADIC: `if n_colors != 4: raise ValueError("Tetradic palette is defined for n_colors=4.")`
    - それ以外のタイプ（Analogous/Complementary/Split Complementary/Tints & Shades）は任意の `n_colors >= 1` に対応している。
  - `src/palette/api.py`
    - `generate_palette(...)` は `generate_raw_colors(...)` → `compute_hue_offsets(...)` を呼び出すため、TRI/TET で不正な `n_colors` を渡すと `ValueError` がスローされる。
- 例外の扱い
  - `_refresh_palette_preview()` / `_update_palette_from_overrides()`（`src/engine/ui/parameters/snapshot.py`）ともに、`build_palette_from_values(...)` 呼び出しを `try/except Exception` で囲んでおり、例外が出ると `palette_obj = None` として黙殺する。
  - その結果、TRI/TET で `n_colors` が 3/4 以外の場合:
    - Palette 生成が失敗 → `palette_obj is None` → プレビューが一切描画されない。
    - `util.palette_state` にも Palette が設定されないため、`api.C` からも色が取得できなくなる（プレビューだけでなくランタイム共有にも影響）。
- GUI 側の前提
  - `src/engine/ui/parameters/manager.py:_register_palette_descriptors`
    - `palette.n_colors` は `RangeHint(2, 6, step=1)` で定義されており、GUI 的には 2〜6 任意値を選べる。
    - TRI/TET 専用の制約や自動補正はここでは行っていない。

→ まとめ: Palette コアが TRI/TET で `n_colors` を 3/4 固定で要求している一方、Parameter GUI では 2〜6 を自由に選べる設計になっており、`build_palette_from_values` がその差分を吸収していないため、TRI/TET + 不適切な `n_colors` の組み合わせで例外 → プレビュー消失が発生している。

## 対応方針（概要）

- Palette コア（`src/palette/harmony.py`）の「TRI/TET は 3/4 固定」という設計は維持しつつ、Parameter GUI 側のアダプタ層で `n_colors` を自動的に有効値に正規化する。
- 具体的には:
  - `engine.ui.palette.helpers.build_palette_from_values` 内で、`PaletteType.TRIADIC` および `PaletteType.TETRADIC` に対して `n_colors` を 3/4 に強制する小さな正規化ロジックを追加する。
  - これにより GUI からは「Colors スライダは 2〜6 のままだが、TRI/TET 選択時には内部的に 3/4 で Palette が生成される」挙動になる。
- 影響範囲:
  - Parameter GUI プレビュー（`_refresh_palette_preview`）: TRI/TET でも常にスウォッチが出るようになる。
  - Snapshot 経由の Palette 更新（`_update_palette_from_overrides`）: TRI/TET + 不正 `n_colors` でも例外が出ず、`api.C` が常に有効な Palette を参照できるようになる。
  - Palette コア API（`palette.generate_palette` 等）の挙動は変更しない（外部から直接呼び出す場合の ValueError 仕様は維持）。

## やること（チェックリスト）

### 1. コア原因の確認とテスト観点整理

- [x] `palette.generate_palette` + `PaletteType.TRIADIC/TETRADIC` + 不正 `n_colors`（例: TRI で 4, TET で 3）を直接呼び出す簡単なテスト/スニペットで `ValueError` 発生を確認する。
- [ ] `engine.ui.palette.helpers.build_palette_from_values` に対して、TRI/TET + 不正 `n_colors` で例外が発生し、`None` が返される現状挙動をテストコードで再現する（ユニットテスト用の最小ケースを決める）。

### 2. `build_palette_from_values` での n_colors 正規化

- [x] `engine.ui.palette.helpers` に `PaletteType` と `n_colors` を受けて有効値を返すヘルパ関数を追加する（例: `_normalize_n_colors_for_type(palette_type, n_colors) -> int`）。
  - TRIADIC の場合: `n_colors` が 3 以外なら 3 に強制。
  - TETRADIC の場合: `n_colors` が 4 以外なら 4 に強制。
  - その他のタイプはそのまま返す。
- [x] `build_palette_from_values` 内で `palette_type` 決定後、このヘルパを呼び出して `n_colors` を正規化し、その値を `generate_palette(...)` に渡すように変更する。
- [x] `build_palette_from_values` の docstring に「一部 PaletteType（TRI/TET）では内部的に `n_colors` を定義値（3/4）に正規化する」旨を追記する。

### 3. Parameter GUI 側との整合性（UI 表示とのギャップ取り扱い）

- [x] Colors スライダ (`palette.n_colors`) の RangeHint は現状の 2〜6 のままとし、まずは内部正規化のみでバグを解消する。
- [ ] TRI/TET 選択時に `palette.n_colors` が 3/4 以外でも内部で 3/4 に正規化されることを、デバッグログなど最低限の形で確認できるようにする（必要なら `engine.ui.palette.helpers` 側で `logger.debug` を追加）。
- [ ] 将来対応案として:
  - TRI/TET 選択時に `palette.n_colors` スライダの値を自動的に 3/4 へ書き戻すかどうかを検討する（ParameterStore の値更新をどこで行うか設計する）。
  - ただし今回の修正スコープでは「内部正規化のみ」とし、UI の値はそのままにしておく。

### 4. スナップショット経路の確認

- [ ] `src/engine/ui/parameters/snapshot.py:_update_palette_from_overrides` に対して、TRI/TET + 不正 `n_colors` を含む overrides を与えたときに:
  - 現状: `build_palette_from_values` からの例外 → `_set_palette(None)` で Palette が消える。
  - 修正後: 例外無しに Palette が生成され、`util.palette_state.get_palette()` から有効な Palette が取得できる。
  を確認するテストケースを追加する。

### 5. 自動テスト追加

- [x] `tests/test_palette_api.py` もしくは新規テストファイル（例: `tests/ui/test_palette_helpers.py`）に以下のテストを追加:
-  - TRIADIC:
-    - `build_palette_from_values` に `palette_type_value="TRI"`, `n_colors_value=2` や 5 などを渡しても例外が出ず、戻り値の `palette.colors` 長が 3 になること。
-  - TETRADIC:
-    - 同様に `palette_type_value="TET"`, `n_colors_value=2` や 5 などを渡しても例外が出ず、`palette.colors` 長が 4 になること。
-  - その他のタイプ（ANA/COM/SPL/TAS）については現状の柔軟な `n_colors` 振る舞いが変わっていないこと（必要ならスモークテスト程度）。
- [x] 既存テストが新しい挙動で失敗しないことを確認し、必要なら期待値を調整する（現状 TRI/TET を直接テストしていないため影響は限定的な想定）。

### 6. 手動確認（Parameter GUI）

- [ ] 実行環境で Parameter GUI を起動し、Palette セクションにて:
  - ANA/COM/SPL/TAS でプレビューが従来どおり動作すること。
  - TRI/TET へ切り替えた際に、`palette.n_colors` が 2〜6 のいずれであってもプレビューのスウォッチが必ず表示されること。
  - `palette.n_colors` を変更してもエラーやプレビュー消失が起きないことを確認する。
- [ ] TRI/TET 選択時に `api.C` から色を参照するスケッチ（例: `C[0]` で線色を決める）を動かし、描画が問題なく行われることを確認する。

### 7. ドキュメント・メモの反映（必要に応じて）

- [ ] 本計画ファイルに実装後の状態を反映し、完了済み項目にチェックを付ける。
- [ ] 必要であれば `palette_integration_plan.md` に「TRI/TET の `n_colors` が GUI から指定された場合の正規化ポリシー」を一文で追記する。
- [ ] 外部利用を想定する場合は `palette/USAGE.md` に「TRI/TET は `n_colors=3/4` 固定であり、他の値を渡した場合は例外が出る」旨を明示し、GUI 側でのみ内部正規化していることを説明するかどうか検討する。

---

この計画に問題なければ、`build_palette_from_values` への `n_colors` 正規化ロジック追加と関連テストの実装から着手する想定です。***
