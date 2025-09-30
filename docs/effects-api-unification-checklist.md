# effects API/実装 統一チェックリスト（提案）

対象: `src/effects/` 配下の純関数エフェクト群（Geometry -> Geometry）

このチェックリストは、引数と実装形式の統一を段階的に行うための作業計画です。コード修正は本チェックリストの承認後に行います。

---

## 現状の主な相違点（要約）
- 角度の単位が混在
  - `rotate/affine/repeat/fill`: ラジアン（`angles_rad`/`angle_rad`）。
  - `twist`: 度（`angle`）。
- 整数パラメータの型が混在
  - `extrude.subdivisions`/`collapse.subdivisions`/`weave.num_candidate_lines`/`weave.relaxation_iterations` が float 受け取り（実際は整数処理）。
- docstring 形式のばらつき（effects/AGENTS.md の規約準拠度）
  - 関数 docstring に Returns/Notes を含むものがある（例: `scale`, `explode`, `collapse` など）。
  - Parameters セクションが省略のものがある（例: `translate`）。
- Numba 利用フラグの名称が不統一
  - `dash`: `PXD_USE_NUMBA_DASH`、`collapse`: `PYX_USE_NUMBA`。`fill` は無条件で numba 使用の補助関数あり。
- パラメータ UI/量子化メタ（`__param_meta__`）の粒度ばらつき
  - `integer` には概ね `step: 1` があるが、float で `step` 未指定が多い（既定 1e-6 に依存）。
- 早期リターン/空入力の扱い記述の差
  - 実質は概ね統一（空はコピー返却）だが、docstring に記述があるもの／ないものが混在。

---

## 統一ルール（合意後に適用）
- 署名・入出力
  - [ ] 関数署名は `def name(g: Geometry, *, ...) -> Geometry` に統一（既定: keyword-only）。
  - [ ] 空入力/無効果（no-op）時は「入力コピーを返す」を明記し、実装も統一。
  - [ ] 出力 dtype は `coords=float32`, `offsets=int32` を堅持。
- 単位・型
  - [ ] 角度はすべてラジアンに統一。名称は `angle_rad`（単体）/`angles_rad`（Vec3）を用いる。
  - [ ] 度を使う現行 API（`twist.angle`）は破壊的変更で `angle_rad` に改名（度→ラジアン変換）。
  - [ ] 実質整数の引数はシグネチャを `int` に統一（`subdivisions`, `count`, `num_candidate_lines`, `relaxation_iterations` 等）。
- ベクトル引数
  - [ ] `Vec3` 受け取りを基本とし、必要に応じて `float|Vec3` を許容し単一値は全成分へ拡張（`ensure_vec3` を使用）。
- docstring（effects/AGENTS.md 準拠）
  - [ ] 関数 docstring は「先頭1行の要約 + Parameters」のみ（Returns/Notes/Examples/実装詳細は不可）。
  - [ ] 単位/レンジ/no-op 条件を Parameters に含める。`__param_meta__` と整合させる。
  - [ ] 詳細な実装メモ/注意点はモジュール docstring へ（必要最小限）。
- `__param_meta__`
  - [ ] すべての公開引数に対して `type`/`min`/`max`/`choices`/`step` を適切に設定。
  - [ ] `integer` には `step: 1`。float の `step` は未指定なら既定（1e-6）に依存で可。UI 上の使い勝手改善が必要なもののみ明示。
- Numba フラグ
  - [ ] 環境フラグ名は `PYX_USE_NUMBA` に統一（効果ごとの粒度が必要な場合は `PYX_USE_NUMBA_<NAME>`）。
  - [ ] numba 未導入環境では確実に NumPy フォールバックし、出力は同一になることを維持。

---

## 実施タスク（ファイル別 TODO）
- affine.py
  - [ ] 関数 docstring を規約形式に整形（Returns/Notes をモジュール側へ集約）。
  - [ ] `__param_meta__` step の要否を点検（現状のままでも可）。
- boldify.py
  - [ ] 関数 docstring を規約形式に整形（Parameters のみに）。
  - [ ] `__param_meta__` は現状維持（boldness のレンジ妥当）。
- collapse.py
  - [ ] 関数 docstring を規約形式に整形（Notes はモジュールへ）。
  - [ ] `subdivisions: float -> int` に変更し、`__param_meta__` は `integer, step:1` を堅持。
  - [ ] Numba フラグを `PYX_USE_NUMBA` に統一（既存と一致、記述の明確化のみ）。
- dash.py
  - [ ] 関数 docstring を規約形式に整形（Parameters のみに）。
  - [ ] Numba フラグを `PXD_USE_NUMBA_DASH -> PYX_USE_NUMBA_DASH` へ改名（または `PYX_USE_NUMBA` へ集約）。
- displace.py
  - [ ] 関数 docstring に Parameters を明記（単位, レンジ, no-op 条件）。
  - [ ] `spatial_freq` は `float|Vec3` の整理を明文化（`ensure_vec3` 準拠へ寄せる検討）。
- explode.py
  - [ ] 関数 docstring を規約形式に整形（Returns/注意の移動）。
  - [ ] `__param_meta__` は現状維持（distance/factor の step は任意）。
- extrude.py
  - [ ] `subdivisions: float -> int` に変更し、`__param_meta__` を `integer, step:1` に統一。
  - [ ] 関数 docstring の Parameters を規約に沿って簡潔化。
- fill.py
  - [ ] 関数 docstring を規約形式に整形（Parameters のみに）。
  - [ ] `angle_rad` はラジアン明記のまま。`density` の UI 用 step の要否検討。
  - [ ] numba 使用箇所のフラグ化（任意: `PYX_USE_NUMBA` で無効化可能に）。
- offset.py
  - [ ] 関数 docstring を規約形式に整形（Parameters のみに）。
  - [ ] `__param_meta__` の `segments_per_circle` は `integer, step:1` を明示（現状OK）。
- repeat.py
  - [ ] 関数 docstring を規約形式に整形。
  - [ ] 角度は `angles_rad_step`（ラジアン）で現状維持。
- rotate.py
  - [ ] 現状ほぼ準拠。docstring の簡潔化のみ確認。
- scale.py
  - [ ] 関数 docstring から Returns を削除（Parameters のみに）。
  - [ ] `scale` 引数（Vec3）に関する `__param_meta__` の step 要否検討。
- subdivide.py
  - [ ] 現状ほぼ準拠。docstring を規約の簡潔さに合わせて微調整。
- translate.py
  - [ ] 関数 docstring に Parameters を追加（`delta` 単位/レンジ/no-op）。
- trim.py
  - [ ] 現状ほぼ準拠。`start_param/end_param` の説明を簡潔に維持。
- twist.py
  - [ ] 角度を度 -> ラジアンへ破壊的変更（`angle -> angle_rad`）。実装で `math.radians` を除去し、直接ラジアンを使用。
  - [ ] `__param_meta__` を `0..2π` レンジに変更。
  - [ ] 関数 docstring を規約形式に整形。
- weave.py
  - [ ] `num_candidate_lines: float -> int`, `relaxation_iterations: float -> int` に変更し、`__param_meta__` は `integer, step:1` を堅持。
  - [ ] 関数 docstring を規約形式に整形。

---

## 追加の検討事項（任意・相談）
- `__param_meta__` の `step` を UI 操作性の観点で明示するか
  - 例: `distance/boldness` は 0.1mm, 角度は ~1e-3rad, `density` は 1.0 など。
  - 署名量子化にも関わるため、キャッシュ鍵の差分が増え過ぎないステップ幅を検討。
- Numba フラグの粒度
  - 全体フラグ `PYX_USE_NUMBA` に一本化するか、効果別 `PYX_USE_NUMBA_<NAME>` を併用するか。
- 破壊的変更の連鎖
  - `twist` の角度単位変更は互換性に影響。未配布リポのため許容方針（AGENTS.md）を踏まえつつ、サンプルやテストの更新も同時に実施。

---

## 実施順（提案）
1) 角度単位/整数型の統一（破壊的変更）
2) docstring 整形（関数: Parameters のみに）
3) `__param_meta__` の整合（type/min/max/choices/step）
4) Numba フラグ名の統一とフォールバック確認

同意いただければ、この順序で最小差分かつ一貫したスタイルで修正を進めます。

---

更新履歴
- 2025-09-30 初版（レビュー結果に基づく草案）
