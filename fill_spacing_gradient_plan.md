# fill spacing_gradient 機能追加計画

塗りつぶしエフェクト `effects.fill.fill` に、スキャン方向に沿って線間隔を徐々に変化させるためのパラメータ `spacing_gradient` を追加する計画とチェックリスト。

## 方針メモ

- パラメータ名は `spacing_gradient` を採用する。
- `spacing_gradient = 0.0` で現状と同じ一様間隔。
- `spacing_gradient > 0` でスキャン軸の「+側」に行くほど線間隔が広がる。
- `spacing_gradient < 0` でスキャン軸の「-側」に行くほど線間隔が広がる。
- 実際の倍率（最後の線間隔 / 最初の線間隔）は `abs(spacing_gradient)` を使って内部で決定する（線形 or 指数カーブの詳細は後で確定）。
- 既存の `density` / `angle_rad` / `angle_sets` の意味は変えない。
- 配列指定は他パラメータと同様に「グループごとにサイクル適用」のルールに合わせる。

## 実装タスクチェックリスト

- [ ] 仕様詳細を確定する
  - [ ] `spacing_gradient` のレンジ（推奨範囲）とデフォルト値（0.0）の妥当性を確認する。
  - [ ] 「実倍率」をどう決めるか（線形: `ratio = 1 + k * abs(spacing_gradient)` か、指数: `ratio = exp(k * abs(spacing_gradient))` など）を決める。
  - [ ] スキャン軸方向の定義（どちらを「+側」とみなすか）を明文化する。
  - [ ] `angle_sets > 1` のとき、各方向ごとの挙動（各方向で独立に gradient を適用する）を明示する。

- [ ] パラメータ定義・メタ情報を更新する
  - [ ] `src/effects/fill.py` の `fill` シグネチャに `spacing_gradient: float | list[float] | tuple[float, ...] = 0.0` を追加する。
  - [ ] `PARAM_META` に `spacing_gradient` の RangeHint を追加する（型: `number`、min/max/step を決める）。
  - [ ] `fill.__param_meta__` が新しい `PARAM_META` を反映することを確認する。
  - [ ] `effects_arguments.md`（必要であれば）に `spacing_gradient` の説明を追記する。

- [ ] 共平面経路のロジックを拡張する
  - [ ] 共平面パスの density/angle/angle_sets の正規化と同様に `spacing_gradient` のシーケンスを正規化する。
  - [ ] `_spacing_from_height` で求めた基準 `spacing` と `spacing_gradient` から、ストライプごとの実 spacing を計算するロジックを設計する。
  - [ ] `_generate_line_fill_evenodd_multi` への入力／内部処理を見直し、必要であれば「ストライプごとの spacing を渡せる」形に拡張するか、小さな補助関数を追加する。
  - [ ] `spacing_gradient = 0.0` のとき、余計な計算を避けて現状と同等のコスト・挙動になるようにする。

- [ ] 非共平面フォールバック経路を拡張する
  - [ ] 非共平面パス（各ポリゴン個別処理）の density/angle/angle_sets 正規化と同様に `spacing_gradient` を適用する場所を決める。
  - [ ] `_fill_single_polygon` で spacing を決める部分に gradient を反映させるか、共平面経路と整合するように共通化する。

- [ ] API / ドキュメントを更新する
  - [ ] `fill` の docstring に `spacing_gradient` の説明を追記する（用途、符号の意味、0.0 との関係）。
  - [ ] 必要に応じて `docs/effects_arguments.md` にパラメータ説明を追加する。
  - [ ] `architecture.md` に effects パラメータ周りの仕様との差分があれば更新する（該当コード参照を添える）。

- [ ] テストと動作確認を行う
  - [ ] 代表的な図形に対して `spacing_gradient = 0.0` の出力が現状と一致することを確認する（既存テストまたは ad-hoc な比較）。
  - [ ] `spacing_gradient > 0` / `< 0` で期待通り「片側に行くほど線間隔が広がる」ことを目視確認する（簡単なスケッチで良い）。
  - [ ] 必要に応じて `tests/` に簡易テストを追加する（例えば、生成線本数や bounding box 上の平均間隔が単調変化しているかをチェック）。
  - [ ] 変更ファイルに対して `ruff` / `mypy` / `pytest`（対象テスト）を実行する。

## 事前に確認したい点・追加提案メモ

- [ ] `spacing_gradient` のレンジ・スケール感
  - 例: `-2.0..2.0` 程度で十分か、もっと極端な値も許容したいか。
  - UI で扱いやすいステップ（0.1 単位など）をどうするか。
- [ ] 実倍率カーブの好み
  - 線形にするか、目視上自然な変化を優先して指数カーブなどを使うか。
  - 単純さを優先してバージョン 1 では線形に固定し、必要があれば `curve_power` などを別パラメータで追加する案も考えられる。
- [ ] angle_sets > 1 のときの直感
  - すべての方向に同じ gradient を適用するか、将来的に「方向ごとに違う gradient」を指定したくなる可能性がどの程度ありそうか。

このチェックリスト・方針で問題なければ、あなたの確認後に実装を進め、完了した項目にチェックを入れていきます。

