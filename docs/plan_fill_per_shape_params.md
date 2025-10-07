# plan: fill の density/angle を図形ごとに配列適用（サイクル対応）

目的
- `src/effects/fill.py` の `fill()` において、`density` と `angle_rad` が「単一 float」に加え「任意長の list/tuple[float]」を受け付けるようにする。
- 塗りつぶし対象の図形（ポリライン）に対し、与えられた `density`/`angle_rad` を順番に適用。図形数がパラメータ配列より多い場合は各配列長で独立にサイクルする（`density[i % len(density)]` / `angle[i % len(angle)]`）。
- XY 共平面の入力では「穴（ホール）」を維持するため、外環＋内環（穴）を 1 グループとして偶奇規則の一括ハッチングを行いつつ、グループ単位で `density`/`angle` を割り当てる。
- 既存のスカラ引数（単一 float）指定時の挙動・出力は維持する（後方互換）。

仕様（追加・変更）
- シグネチャ: `fill(g, *, angle_sets: int = 1, angle_rad: float | list[float] | tuple[float, ...] = pi/4, density: float | list[float] | tuple[float, ...] = 35.0, remove_boundary: bool = False)`
  - `angle_rad`/`density` がスカラの場合: 現状通り全図形へ同一値適用。
  - `angle_rad`/`density` が配列の場合: 図形（グループ）順に割り当て、長さに応じて個別にサイクル。
  - `angle_sets` は従来通り有効。各図形に対し `angle = base + i*(pi/k)` を合成。
- GUI/RangeHint:
  - `__param_meta__` は当面 `{"density": {"type": "number"}, "angle_rad": {"type": "number"}}` を維持（スカラの調整 UI 用）。配列指定はコードからのみ利用（UI は非対象）。
- 量子化/署名:
  - 既存の `common.param_utils.quantize_params` は list/tuple に対応済み（成分ごと量子化→tuple 化）。追加対応は不要。

実装タスク（チェックリスト）
- [ ] `fill()` の型注釈と docstring を更新（配列受け入れとサイクリング仕様を明記）。
- [ ] `density`/`angle_rad` の正規化ヘルパを追加（`_normalize_scalar_or_seq(x) -> list[float]`）。
- [ ] XY 共平面パスのグルーピング実装:
  - [ ] 各ポリラインの 2D 重心点を計算。
  - [ ] `point_in_polygon_njit` を用いて包含関係（他ポリゴンに内包される数）を算出し、偶奇で外環/内環を判定。
  - [ ] 各外環ごとに直下の内包リング（穴）を集めて 1 グループ化。
  - [ ] グループ順（入力出現順の外環）に対し `density_seq[i%Ld]`/`angle_seq[i%La]` を割当。
  - [ ] 既存の `_generate_line_fill_evenodd_multi` をグループ単位で呼び出し、`angle_sets` を適用して合成。
  - [ ] `remove_boundary` の挙動を維持（False なら境界線を先頭に残す）。
- [ ] 非平面パスの適用順序変更:
  - [ ] 各ポリラインを個別処理し、インデックスに基づき `density`/`angle` をサイクル割当。
  - [ ] 既存の平面性チェックとスキップ条件は維持。
- [ ] 既存の `__param_meta__` は数値のまま据え置き。必要に応じ docstring に GUI 非対応注記を追記。
- [ ] スタブ生成の型表現更新:
  - [ ] `tools/gen_g_stubs._annotation_for_effect_param` を拡張し、`tuple[float, ...]` を正しく `tuple[float, ...]` として描画（現在の `Ellipsis` 表示を修正）。
  - [ ] `list[float]` はそのまま描画。追加 import は不要（`list[...]` は組込み）
  - [ ] 生成後、`src/api/__init__.pyi` の `fill` の引数型が `float | list[float] | tuple[float, ...]` になっていることを確認。

テスト計画（変更ファイル優先）
- 追加テスト（`tests/effects/test_fill_per_shape_params.py`）
  - [ ] 平面・離散 3 図形: `density=[5,10,20]`, `angle=[0, pi/2]`, `angle_sets=1`
    - 各図形の重心に対して出力線の中点が内包される線分を集計し、本数がおおよそ `density` に一致すること。
    - 角度は各図形で交差角が異なること（方向ベクトルの atan2 で比較）。
  - [ ] サイクル動作: 5 図形に対し `density=[3,7]`, `angle=[0]` で 1,3,5 番目が 3、2,4 番目が 7 付近の本数になること。
  - [ ] 非平面 2 図形: 各図形に別々の `density/angle` が適用されること（境界保持含む）。
  - [ ] 後方互換: スカラ指定時に既存テスト（`tests/test_effect_fill_*`）がすべて緑のままであること。
- 検証手順
  - ruff/black/isort/mypy: 変更ファイルに限定して実行。
  - pytest: 追加テストと既存の `tests/test_effect_fill_*` を優先実行。
  - スタブ同期: `python -m tools.gen_g_stubs && git add src/api/__init__.pyi` → `pytest -q tests/stubs/test_g_stub_sync.py`。

互換性・リスク・方針
- 後方互換: 既存コードはスカラ指定を継続利用可能。デフォルト値・既存メタは不変。
- 穴の維持: XY 共平面では外環＋穴のグループ単位で偶奇処理するため、配列適用でも穴は維持される。
- UI との整合: 配列入力はプログラマ向け機能。GUI スライダはスカラのみを対象とし、`__param_meta__` は据え置く。
- 計算量: グルーピングは O(n^2)（包含判定）だが n は小さい前提で許容。
- スタブ生成: 可変長タプルの表記修正は生成器側の小変更で対応。

実行順序（サマリ）
1) 型注釈/正規化/サイクル割当の実装
2) 平面グルーピング＋偶奇合成の実装
3) 非平面パスの割当
4) テスト追加と最小限の既存テスト確認
5) スタブ生成器更新→スタブ再生成→スタブ同期テスト緑化
6) ドキュメント微修正（必要箇所）

承認依頼
- 上記方針で実装を進めてよいか確認してください。特に「XY 共平面での外環＋穴のグループ単位適用」方針と、`__param_meta__` をスカラ維持（GUI 非対応）の扱いで問題ないかご確認ください。
