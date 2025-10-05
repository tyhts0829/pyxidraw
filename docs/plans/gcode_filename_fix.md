# G-code ファイル名にキャンバス寸法とスクリプト名を反映する修正計画

目的: `G` 押下で保存される G-code の既定ファイル名から `unknownWxH` を排し、寸法とエントリスクリプト名を反映した命名に揃える。

背景/現状
- 現状は `engine.export.service._make_gcode_filename()` が固定で `unknownWxH_mm` を付与している。
  - 該当: `src/engine/export/service.py:231` 付近（`base = f"{ts}_unknownWxH_mm"`）。
- ランナー側では `canvas_height_mm` を `GCodeParams` に渡しているが、ファイル名生成では未使用。
  - 参照: `src/api/sketch.py:410`（`GCodeParams(y_down=True, canvas_height_mm=float(canvas_height))`）。
- 既定仕様では `YYYYmmdd_HHMMSS_{W}x{H}_mm.gcode` が期待値。
  - 参照: `docs/export_image_spec.md:24`。

スコープ/非スコープ
- スコープ: 既定ファイル名の決定ロジックのみ（保存先や書き出し本体の変更なし）。
- 非スコープ: G-code 生成ロジック、HUD/UI の振る舞い、実 Writer 実装の拡張。

命名ルール（提案）
- エントリスクリプト名が取得できる場合（例: `main.py`）:
  - `{script_stem}_{W}x{H}_yymmdd_hhmmss.gcode`
  - 例: `main_400x400_250930_223012.gcode`
- スクリプト名が取得できない場合（または未指定）:
  - `YYYYmmdd_HHMMSS_{W}x{H}_mm.gcode`
- 寸法が取得できない場合は `unknownWxH` を用いる（例: `main_unknownWxH_250930_223012.gcode` / `YYYYmmdd_HHMMSS_unknownWxH_mm.gcode`）。
- 幅・高さの丸めは整数 mm（`int(round(...))`）。

PNG（スクリーンショット）の命名（追加）
- エントリスクリプト名が取得できる場合:
  - `{script_stem}_{W}x{H}_{yymmdd_hhmmss}.png`
  - 例: `main_400x400_250930_223012.png`
- 取得できない場合のフォールバック:
  - `YYYYmmdd_HHMMSS_{W}x{H}.png`（従来はピクセル表記だったが、mm に統一）
- 寸法が取得できない場合は `unknownWxH`（例: `main_unknownWxH_250930_223012.png`）。

方針（設計）
- 寸法とスクリプト名をファイル名に反映する。
  - エントリスクリプト（`sys.argv[0]` の stem）を取得し、利用可能なら先頭に付与。
  - 例: `main.py` からの実行時は `main_{W}x{H}_yymmdd_hhmmss.gcode`。
- ファイル名生成にキャンバス寸法 [mm] を利用できるように、ジョブ投入時の `params`（`GCodeParams`）から `width/height` を取得して使用する。
- `GCodeParams` に `canvas_width_mm: float | None` を追加（`canvas_height_mm` は既存）。
- `src/api/sketch.py` 側で `canvas_width_mm` も渡す（現行と同様に mm 単位）。
- 両寸法が正の数で取得できた場合のみ `{W}x{H}` を付与。欠損時は従来どおり `unknownWxH` を用いる。
- タイムスタンプはスクリプト名ありのとき `yymmdd_hhmmss`、なしのときは現行に近い `YYYYmmdd_HHMMSS` を使用（要確認）。

PNG への適用（設計）
- `engine.export.image.save_png` でファイル名決定に `name_prefix` とキャンバス寸法 [mm] を使えるようにする。
  - 既定では `screenshots/` 配下に `{prefix}_{W}x{H}_{yymmdd_hhmmss}.png`。
  - `path` が明示された場合は従来通りそれを優先。
  - `include_overlay`/`scale` の指定にかかわらず、W/H はキャンバスの mm を用いる（スケールはピクセル出力にのみ影響）。

実装手順（チェックリスト）
- [x] `engine.export.gcode.GCodeParams` に `canvas_width_mm: float | None = None` を追加し、docstring を更新。
- [x] `api/sketch.py` の G-code 実行箇所で `GCodeParams(canvas_width_mm=float(canvas_width), canvas_height_mm=float(canvas_height))` を渡す。
- [x] `ExportService.submit_gcode_job(...)` に `name_prefix: str | None = None`（kw-only）を追加。
- [x] `engine.export.service._make_gcode_filename(out_dir: Path, width_mm: float | None, height_mm: float | None, name_prefix: str | None, ts_style: Literal["yymmdd","yyyymmdd"] = "yymmdd")` に変更し、
      - [x] `name_prefix` があれば `{name_prefix}_{W}x{H}_{yymmdd_hhmmss}.gcode` を優先（W/H 不明時は `{name_prefix}_unknownWxH_{yymmdd_hhmmss}.gcode`）。
      - [x] `name_prefix` が無い場合は従来形式に近い `YYYYmmdd_HHMMSS_{W}x{H}_mm.gcode` を維持（W/H 不明時は `unknownWxH_mm`）。
- [x] `ExportService._run_job` 内で `job.params` から幅高を取得し、`name_prefix` は `Path(sys.argv[0]).stem` を `api/sketch.py` から渡して `_make_gcode_filename` に供給。
- [x] 既存の `_unique_path` により衝突回避（`-1`, `-2`, ...）が継続することを確認。

PNG 側の実装（チェックリスト）
- [x] `engine.export.image.save_png` に `name_prefix: str | None = None, width_mm: float | None = None, height_mm: float | None = None` を追加。
- [x] `path is None` の場合の自動命名を `{prefix}_{W}x{H}_{yymmdd_hhmmss}.png`（prefix なし時は `YYYYmmdd_HHMMSS_{W}x{H}.png`）に変更。
- [x] `src/api/sketch.py` の `on_key_press`（P/Shift+P）呼び出しで、`name_prefix=Path(sys.argv[0]).stem, width_mm=canvas_width, height_mm=canvas_height` を渡す。

テスト/検証（編集ファイル優先）
- [ ] 変更ファイルに限定して `ruff/black/isort/mypy` を実行。
- [ ] 既存スモーク `tests/ui/test_export_minimal.py::test_export_simulated_complete` が通ることを確認。
- [ ] 追加: `tests/ui/test_export_with_writer.py` を拡張し、`name_prefix="main"` と寸法ありのときに `main_{W}x{H}_yymmdd_hhmmss.gcode` が生成されることを文字列パターンで検証。
- [ ] （任意）PNG: ヘッドレス環境を考慮し、ユニットでは命名ユーティリティを分離して文字列生成のみ検証、もしくは小さな関数テストを追加。

リスクと互換性
- `GCodeParams` の後方互換: 新規フィールドはデフォルト `None` のため既存呼び出しに影響なし。
- 生成ファイル名の変更: 既存の `unknownWxH_mm` を期待するワークフローがあれば影響する可能性はあるが、現状ユーザー配布前で影響は限定的。

要確認事項（ご指示ください）
1) 寸法の丸め: 整数 mm で問題ないか（小数を許容する場合、`{W:.1f}x{H:.1f}` 等に変更可能）。
2) 単位サフィックス: 例では `_mm` が無いため、`main_{W}x{H}_yymmdd_hhmmss.gcode`（単位表記なし）で確定か。既定（プレフィックスなし）パスでは `_mm` を残すか除くか。
3) 片方のみ取得できた場合の挙動: `unknownWxH` へフォールバックでよいか。
4) スクリプト名の汎用化: `main.py` に限らず、任意スクリプト（例: `foo.py`）でも `{stem}_{W}x{H}_yymmdd_hhmmss.gcode` とするか。
5) 既定の原点/スケールの扱い: 本変更はファイル名のみで、実座標系（原点/スケール）は不変更でよいか。
6) PNG 側も単位表記なしで統一（`.png` は `{W}x{H}` のみ）でよいか。

完了条件（DoD）
- [ ] `G` 押下で保存された G-code のファイル名が、スクリプト名が分かる場合は `{script}_{W}x{H}_yymmdd_hhmmss.gcode`。
- [ ] スクリプト名が不明な場合も `YYYYmmdd_HHMMSS_{W}x{H}_mm.gcode` など定義済みフォールバックに自動切替。
- [ ] 寸法不明時は `unknownWxH` フォールバック。
- [ ] 変更ファイルの `ruff/mypy/pytest -q -m smoke` が成功。
