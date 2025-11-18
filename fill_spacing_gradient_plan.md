# fill spacing_gradient 機能追加計画

塗りつぶしエフェクト `effects.fill.fill` に、スキャン方向に沿って線間隔を徐々に変化させるためのパラメータ `spacing_gradient` を追加する計画とチェックリスト。

## 方針メモ

- パラメータ名は `spacing_gradient` を採用する。
- `spacing_gradient = 0.0` で現状と同じ一様間隔。
- `spacing_gradient > 0` でスキャン軸の「+側」に行くほど線間隔が広がる。
- `spacing_gradient < 0` でスキャン軸の「-側」に行くほど線間隔が広がる。
- 実際の倍率（最後の線間隔 / 最初の線間隔）は `exp(abs(spacing_gradient))` で決定する（指数カーブ）。0.0 付近ではほぼ線形に近い変化。
- 既存の `density` / `angle_rad` / `angle_sets` の意味は変えない。
- 配列指定は他パラメータと同様に「グループごとにサイクル適用」のルールに合わせる。

## 実装タスクチェックリスト

- [x] 仕様詳細を確定する
  - [x] `spacing_gradient` のレンジ（推奨範囲）とデフォルト値（0.0）の妥当性を確認する。RangeHint は `[-2.0, 2.0]`、step は `0.1` とし、内部では `[-4.0, 4.0]` にソフトクランプする。
  - [x] 「実倍率」をどう決めるか（線形: `ratio = 1 + k * abs(spacing_gradient)` か、指数: `ratio = exp(k * abs(spacing_gradient))` など）を決める。指数カーブを採用し、`ratio = exp(abs(spacing_gradient))` となるように実装する。
  - [x] スキャン軸方向の定義（どちらを「+側」とみなすか）を明文化する。XY（または回転後作業座標）の Y 軸正方向を「+側」とし、`spacing_gradient > 0` でその方向へ行くほど線間隔が広がる。
  - [x] `angle_sets > 1` のとき、各方向ごとの挙動（各方向で独立に gradient を適用する）を明示する。各方向のスキャン軸ごとに同じ `spacing_gradient` を適用する。

- [x] パラメータ定義・メタ情報を更新する
  - [x] `src/effects/fill.py` の `fill` シグネチャに `spacing_gradient: float | list[float] | tuple[float, ...] = 0.0` を追加する。
  - [x] `PARAM_META` に `spacing_gradient` の RangeHint を追加する（型: `number`、min/max/step を決める）。
  - [x] `fill.__param_meta__` が新しい `PARAM_META` を反映することを確認する。
  - [x] `effects_arguments.md`（必要であれば）に `spacing_gradient` の説明を追記する。

- [x] 共平面経路のロジックを拡張する
  - [x] 共平面パスの density/angle/angle_sets の正規化と同様に `spacing_gradient` のシーケンスを正規化する。
  - [x] `_spacing_from_height` で求めた基準 `spacing` と `spacing_gradient` から、ストライプごとの実 spacing を計算するロジックを設計する（`_generate_y_values` ヘルパー関数を追加）。
  - [x] `_generate_line_fill_evenodd_multi` への入力／内部処理を見直し、「ストライプごとの spacing を渡せる」形に拡張する（`spacing_gradient` 引数を追加し、内部で `_generate_y_values` を利用）。
  - [x] `spacing_gradient = 0.0` のとき、余計な計算を避けて現状と同等のコスト・挙動になるようにする（`_generate_y_values` 内で早期に一様 `np.arange` を返す）。

- [x] 非共平面フォールバック経路を拡張する
  - [x] 非共平面パス（各ポリゴン個別処理）の density/angle/angle_sets 正規化と同様に `spacing_gradient` を適用する場所を決める（ポリゴン単位でサイクル適用）。
  - [x] `_fill_single_polygon` で spacing を決める部分に gradient を反映させるか、共平面経路と整合するように共通化する（`_generate_line_fill` に `spacing_gradient` を渡す）。

- [x] API / ドキュメントを更新する
  - [x] `fill` の docstring に `spacing_gradient` の説明を追記する（用途、符号の意味、0.0 との関係）。
  - [x] 必要に応じて `docs/effects_arguments.md` にパラメータ説明を追加する。
  - [x] `architecture.md` に effects パラメータ周りの仕様との差分があれば更新する（該当コード参照を添える）。今回の変更では追記不要と判断。

- [x] テストと動作確認を行う
  - [x] 代表的な図形に対して `spacing_gradient = 0.0` の出力が現状と一致することを確認する（既存テストまたは ad-hoc な比較）。既存の fill 関連テスト（`tests/effects/test_fill_angle_sets_cycle.py`, `tests/effects/test_fill_per_shape_params.py`）が引き続きパスすることを確認。
  - [ ] `spacing_gradient > 0` / `< 0` で期待通り「片側に行くほど線間隔が広がる」ことを目視確認する（簡単なスケッチで良い）。
  - [ ] 必要に応じて `tests/` に簡易テストを追加する（例えば、生成線本数や bounding box 上の平均間隔が単調変化しているかをチェック）。
  - [x] 変更ファイルに対して `ruff` / `mypy` / `pytest`（対象テスト）を実行する。

## 事前に確認したい点・追加提案メモ

- [x] `spacing_gradient` のレンジ・スケール感
  - RangeHint は `[-2.0, 2.0]`、内部クランプは `[-4.0, 4.0]` とした（`|gradient| ≲ 2.0` を実用的な範囲と想定）。
  - UI で扱いやすいステップとして 0.1 を採用した。
- [x] 実倍率カーブの好み
  - 指数カーブを採用し、`gradient` の絶対値が大きいほど端の間隔比 `exp(|gradient|)` が増加する設計とした。
  - バージョン 1 では `spacing_gradient` のみとし、`curve_power` のような追加パラメータは導入していない。
- [x] angle_sets > 1 のときの直感
  - すべての方向に同じ `spacing_gradient` を適用し、方向ごとの差は今後の拡張余地として残した。

今後は「テストと動作確認を行う」の各項目を進めつつ、必要に応じて追加の調整や改善案をこのファイルに追記する。
