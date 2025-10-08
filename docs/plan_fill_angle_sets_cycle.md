# plan: fill の angle_sets を図形ごとに配列適用（サイクル対応）

目的
- `src/effects/fill.py` の `fill()` において、`angle_sets` を「単一 int」に加え「任意長の list/tuple[int]」で受け付け、
  図形（XY 共平面では外環＋穴グループ）ごとに順番適用。長さに応じてサイクルする（`k_i = angle_sets_seq[i % L]`）。
- 既存のスカラ指定・既定値の動作はそのまま維持（後方互換）。

仕様（追加・変更）
- シグネチャ: `fill(g, *, angle_sets: int | list[int] | tuple[int, ...] = 1, angle_rad: float | list[float] | tuple[float, ...] = pi/4, density: float | list[float] | tuple[float, ...] = 35.0, remove_boundary: bool = False)`
  - `angle_sets` スカラ: 現行と同一（全図形へ同一の方向数）。
  - `angle_sets` 配列: 図形（グループ）順に割り当て、長さに応じてサイクル。
  - 各図形（グループ）で `k=max(1,int(k_i))` を用い、`angle = base + j*(pi/k)`（j=0..k-1）で合成。
- GUI/RangeHint: `__param_meta__['angle_sets']` は integer のまま（UI はスカラのみ操作）。
- 量子化/署名: 既存の `quantize_params` は list/tuple に対応。int は非量子化で既存挙動。

実装タスク（チェックリスト）
- [x] `fill()` の仮引数型注釈と docstring を更新（`angle_sets` の配列対応とサイクリング仕様を追記）。
- [x] `angle_sets` の正規化ヘルパを追加（`_as_int_seq(x) -> list[int]`）。
- [x] XY 共平面パス:
  - [x] 既存のグルーピング `_build_evenodd_groups` の出力インデックスに対し `k = angle_sets_seq[idx % L]` を適用。
  - [x] グループ単位で `density/angle_rad/angle_sets` をサイクル割当し、`_generate_line_fill_evenodd_multi` を `j=0..k-1` で合成。
- [x] 非平面パス:
  - [x] ポリライン i に対し `k = angle_sets_seq[i % L]` を適用し、`_fill_single_polygon` 呼び出し時に渡す。
- [x] `__param_meta__` は据え置き（GUI スカラ）。
- [x] スタブ再生成で `angle_sets: int | list[int] | tuple[int, ...]` が反映されることを確認。

テスト計画（変更ファイル優先）
- 追加テスト `tests/effects/test_fill_angle_sets_cycle.py`
  - [x] 3 つの離散スクエア、`angle_rad=0`、`density` は固定、`angle_sets=[1,2]`、`remove_boundary=True`。
    - [x] 中央（k=2）は横・縦の両方向が含まれる（水平/垂直の本数がいずれも >0）。
    - [x] 左右（k=1）は横方向のみ（垂直がほぼ 0）。
  - [x] 後方互換: 既存 `tests/test_effect_fill_*` が緑のまま。
- 検証
  - ruff/black/isort/mypy（変更ファイル限定）
  - pytest（上記追加テスト＋既存関連）
  - スタブ同期テスト

リスク・方針
- `angle_sets<=0` は 1 に丸める（既存互換）。
- UI はスカラ維持のため、操作系の変更は行わない。

承認依頼
- 上記の方針で実装してよいか確認してください（密度/角度の配列適用と同一ポリシーで `angle_sets` をサイクル適用）。
