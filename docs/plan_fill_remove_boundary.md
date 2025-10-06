# fill: 境界線削除トグル 追加 — 設計/実装計画

目的
- 塗りつぶし効果で、領域を形成する閉じた輪郭線（アウトライン）を最終結果から除去するかを引数で切り替え可能にする。
- 既定値は現挙動（アウトラインを残す）を維持し、後方互換を担保する。

スコープ
- 対象: `src/effects/fill.py` の `fill()`（および内部ヘルパ）と `__param_meta__`、公開スタブ更新、最小限のドキュメント/テスト追加。
- 非対象: 塗りつぶしアルゴリズム自体の最適化/挙動変更、他エフェクトへの波及。

仕様（提案）
- 追加引数: `remove_boundary: bool = False`
  - 意味: True のとき、塗りつぶし対象領域の元の閉ループ線を出力に含めない（除去）。False で従来通り含める。
  - 既定値が False の理由: 現挙動（輪郭を残す）との後方互換性維持。
- UI メタ: `fill.__param_meta__` に `{"remove_boundary": {"type": "boolean"}}` を追加（RangeHint は不要）。
- 平面XY最適化パス（偶奇規則）: 元の輪郭追加（`results.append(...)`）を `if not remove_boundary:` でガード。
  - 変更箇所: `src/effects/fill.py:352` 近傍（`results` へ元輪郭を push するループ）。
- 非平面パス: `_fill_single_polygon(...)` に引数を伝播し、`out` 初期化を `[]` または `[vertices]` で分岐。
  - 変更箇所: `src/effects/fill.py:404` シグネチャ拡張と `src/effects/fill.py:416` 近傍の `out` 初期化。
- 密度ゼロ時の扱い（no-op）: 既存仕様維持（density <= 0 では入力をそのまま返す）。`remove_boundary` はこの条件下では効果なし。
- キャッシュ/量子化: 新規 bool パラメータは量子化対象外（既定仕様）。パイプライン鍵には値が反映される（既存の `params_signature` 挙動）。

変更箇所一覧
- `src/effects/fill.py:332` 署名に `remove_boundary: bool = False` を追加し docstring へ説明を追記。
- `src/effects/fill.py:397` の `fill.__param_meta__` に `remove_boundary` を追加。
- `src/effects/fill.py:404` `_fill_single_polygon` のシグネチャに `remove_boundary: bool` を追加し、呼び出し側から引き渡し。
- `docs/pipeline.md` の fill 使用例に引数を軽く言及（任意）。
- `src/api/__init__.pyi` の公開スタブを再生成（自動）して同期。

実装タスクチェックリスト
- [x] `fill(g, *, mode, angle_rad, density, remove_boundary=False)` を追加（`src/effects/fill.py`）。
- [x] 平面XYパス: 元輪郭 push 部分を `if not remove_boundary:` で条件化。
- [x] 非平面パス: `_fill_single_polygon(..., remove_boundary=remove_boundary)` を渡し、`out = [] if remove_boundary else [vertices]` に変更。
- [x] `fill.__param_meta__` に `{"remove_boundary": {"type": "boolean"}}` を追加。
- [x] `fill` の docstring Parameters に `remove_boundary` の説明を追加（一文、NumPy スタイル）。
- [x] 公開スタブ再生成: `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`（差分ゼロを確認）。
- [x] 変更ファイルに限定したフォーマット/型/テスト実行（下記コマンド参照）。
- [ ] `docs/pipeline.md` に 1 行の利用例を追加（任意）。

テスト計画（最小）
- 新規: `tests/test_effect_fill_remove_boundary.py`
  - ケース1（デフォルト維持）: `G.polygon(n_sides=4)` を入力に、`E.pipeline.fill(mode="lines", density=10).build()(g)` の出力に「入力と同一の頂点列を持つ線」が含まれることを `np.array_equal` で確認。
  - ケース2（除去ON）: 同入力で `remove_boundary=True` とし、上記の同一線が含まれないことを確認。総線分数が減少していることも補助的に検証。
  - 縁値: `density<=0` のときは入力がそのまま返る（`remove_boundary` の有無に関わらず同値）ことを確認。
- 実行例（編集ファイル優先）:
  - Lint: `ruff check --fix src/effects/fill.py tests/test_effect_fill_remove_boundary.py`
  - Format: `black src/effects/fill.py tests/test_effect_fill_remove_boundary.py && isort ...`
  - Type: `mypy src/effects/fill.py`
  - Test: `pytest -q tests/test_effect_fill_remove_boundary.py`

互換性/影響
- 後方互換: 既定 `remove_boundary=False` のため既存呼び出しは挙動不変。
- UI: `type: "boolean"` によりトグル表示（明示引数指定時は GUI 側に出さない現仕様を踏襲）。
- 性能: 条件分岐のみでコスト増は無視可能。

オープン質問（要確認）
- パラメータ名: `remove_boundary`（提案）でよいか。代案: `remove_outline`, `keep_outline`（否定形を避けたい場合）。
- `density<=0` のときに「境界のみ除去」モードを許すか（提案は現状維持: no-op）。
- ドキュメント反映の範囲: `docs/pipeline.md` のみ追記で十分か、別途 effects リファレンスを設けるか。

作業コマンド（参考 / 事前承認不要の最小セット）
- フォーマット/型/テスト（変更ファイルのみ）:
  - `ruff check --fix src/effects/fill.py`
  - `black src/effects/fill.py && isort src/effects/fill.py`
  - `mypy src/effects/fill.py`
  - `pytest -q tests/test_effect_fill_remove_boundary.py`
- スタブ再生成（Ask-first なしで可。生成後の差分は PR で提示）:
  - `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

想定コミットメッセージ（参考）
- `feat(effects): add remove_boundary toggle to fill`

完了条件
- 変更ファイルに対する `ruff/black/isort/mypy/pytest` が緑。
- `api/__init__.pyi` のスタブ差分ゼロ（再生成後に同期）。
- 既存の perf テスト（カタログ走査）は影響なし。

備考
- 本変更は純関数性/キャッシュ仕様を保つ。`__param_meta__` と docstring を同時更新する。
- 追加の最適化やアルゴリズム変更は行わない（スコープ外）。
