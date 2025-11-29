# fill エフェクト: 線角度グラデーション引数追加 計画

## ゴール

- `effects.fill` に「線の角度」を調整できる引数（仮称: `angle_gradient`）を追加し、`spacing_gradient` と同様に連続的な変化を表現できるようにする。
- 既存の密度制御・偶奇規則・回転不変性テストを壊さず、シンプルで理解しやすい実装にとどめる。

## TODO: 仕様設計（要確認）

- [ ] 新しい引数名を決定する（案: `angle_gradient`, `angle_rad_gradient`）。`angle_rad` との関係が直感的になる名称にする。
- [ ] 角度グラデーションの定義を決める（例: スキャン軸上の正規化位置 `t` に応じて `angle(t) = angle_rad + f(t) * gradient` の形で変化させる）。
- [ ] gradient の単位とレンジを決める（案: ラジアン単位で `[-pi/4, pi/4]` 程度、または無次元パラメータを `spacing_gradient` と同様のスケールで扱う）。
- [ ] `angle_rad` / `angle_sets` / `spacing_gradient` と組み合わせたときの優先順位・直感的な挙動を明文化する。
- [ ] 共平面経路（多輪郭 + 偶奇）と非共平面フォールバックで同じ仕様になるようにするか、差がある場合は仕様として明示する。
- [ ] Parameter GUI でのラベル・説明文（例: 「角度グラデーション」）と RangeHint（min/max/step）が分かりやすくなるように決める。
- [ ] 新しい float パラメータの量子化ステップ（`__param_meta__['step']`）を決め、キャッシュ署名（`params_signature`）との整合を確認する。

## TODO: API / メタデータ

- [ ] `src/effects/fill.py` の `PARAM_META` に新しい角度グラデーション引数を追加し、`fill.__param_meta__` へ反映する。
- [ ] `_as_float_seq` で新パラメータを受け取れるようにし、現在のエラーメッセージ（`"angle_rad/density/spacing_gradient は ..."`）を新パラメータ名を含む形に整理する。
- [ ] planar / non-planar 双方で新パラメータの配列展開（`*_seq`）を行い、図形（グループ）ごとにサイクル適用できるようにする。
- [ ] `src/api/__init__.pyi` の `fill` シグネチャおよび meta コメント群に新パラメータを追加する（RangeHint と説明文を spacing_gradient と対応づける）。
- [ ] スタブ生成スクリプト `tools/gen_g_stubs.py` を更新し、`python -m tools.gen_g_stubs` 実行で `src/api/__init__.pyi` の定義が自動的に最新化されることを確認する。
- [ ] 公開 API 変更として、`mypy src/effects/fill.py src/api/__init__.pyi` が通ることを確認する。

## TODO: 実装

- [ ] `_generate_line_fill` に新しい角度グラデーション引数を追加し、線分ごとに角度を変化させるロジックを設計する。
- [ ] 交点計算と回転処理の整合性が保たれるように、グラデーション適用のタイミングを決める（例: 作業座標系でのスキャンを維持しつつ、角度変化を「グループ単位」あるいは「スキャン軸上の緩やかな補間」に限定する）。
- [ ] `_generate_line_fill_evenodd_multi` に新パラメータを伝搬し、共平面・多輪郭・偶奇規則ベースの塗りにも同じ制御を適用する。
- [ ] `_fill_single_polygon` から `_generate_line_fill` への呼び出し経路に新パラメータを追加し、非共平面フォールバック経路でも挙動が一致するようにする。
- [ ] `angle_sets` が 1 より大きい場合にも自然な挙動になるように、基準角 `angle_rad` とセット数ごとの角度ずらしとの組み合わせを見直す。
- [ ] 極端な gradient 値で線が過度にねじれたり、塗り領域から大きくはみ出さないように、軽いクランプや安全ガード（NaN/inf チェックなど）を追加する。
- [ ] Numba 最適化関数群（`find_line_intersections_njit`, `generate_line_intersections_batch`）のシグネチャを変更せずに済むか検討し、必要な場合でも変更を最小限にとどめる。

## TODO: テスト / 検証

- [ ] 既存の fill 関連テストがすべて通ることを確認する（例: `tests/test_effect_fill_nonplanar_skip.py`, `tests/test_effect_fill_remove_boundary.py`, `tests/smoke/test_fill_rotation_invariance.py`, `tests/effects/test_fill_*`）。
- [ ] `angle_gradient=0.0`（または既定値）のときに、現在の挙動と完全に互換であることを確認する回帰テストを追加する。
- [ ] 正/負/ゼロの gradient 値、大小の値、list/tuple 指定、`angle_sets>1` の組み合わせをカバーするテストケースを追加する。
- [ ] `spacing_gradient` と新しい角度グラデーションを同時に使用した際の代表ケース（単純な矩形、穴あきポリゴン、非共平面入力など）をテストし、直感的な結果になることを確認する。
- [ ] パフォーマンス退行が許容範囲内かを確認するため、必要に応じて `pytest -q tests/perf/test_catalog_perf.py -k fill` を実行し、実行時間の変化を目視で確認する。

## TODO: ドキュメント / メンテナンス

- [ ] `effects.fill` の docstring に新パラメータの説明を追加し、`spacing_gradient` との関係が分かるようにする。
- [ ] Parameter GUI 周辺のドキュメント（必要であれば `docs/` や `architecture.md`）を確認し、fill パラメータの説明と差分があれば更新する。
- [ ] 変更内容を簡潔に要約したメモ（設計意図・制約・既知のトレードオフ）をこのファイルか別途 ADR に残す。

## メモ / オープンな論点

- 角度グラデーションを「スキャン軸に沿った連続的な角度変化」として実装するか、「図形/グループ単位の角度補間」としてより単純なモデルにするかは要検討（表現力と実装の複雑さのバランス）。
- 交点計算と角度グラデーションの組み合わせで塗り領域から線がはみ出さないようにするため、近似ではなく一貫性のある座標変換パターンを選ぶ必要がある。
- 実際にスケッチで使ってみて、どの程度の gradient レンジが「気持ちよく」感じられるか（例: `[-pi/6, pi/6]` 以内）を確認し、RangeHint を調整したい。
- 最初の実装では仕様をシンプルに保ち、必要であれば将来のバージョンでより高度な角度制御（例: 非線形プロファイル）を検討する。

