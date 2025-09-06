# エフェクト既定値チューニング作業メモ（引き継ぎ用）

目的
- 破壊的変更（2025-09-03）の後、各エフェクトの「既定パラメータ」を見直し、単体プレビューで意図が一目で伝わる値に調整する。
- 判定は「キャンバス中央の立方体にそのエフェクトを1つだけ適用した静止画」で行う。

基本方針
- 既定値は「効果が分かる最小限の強さ」かつ「形状の破綻・画面外逸脱を避ける」バランスを優先。
- 既定値変更はアルゴリズムの意味を変えない範囲（デフォルト引数値と docstring 追記、必要なら軽微な可視化修正）。
- `__param_meta__` の整合を保つ（validate_spec が通ること）。

プレビュー・再現手順
- ヘッドレス・スナップショット（OpenGL不要・Matplotlib Agg）
  - 個別: `python scripts/tune_<effect>.py --help`
    - 例: `python scripts/tune_twist.py --angle 60 --out screenshots/twist.png`
          `python scripts/tune_fill.py --mode lines --density 0.35 --angle 0.7854 --out screenshots/fill.png`
          `python scripts/tune_dash.py --dash_length 6 --gap_length 3 --out screenshots/dash_default.png`
  - グリッド（参考）: `python scripts/snapshot_effects_grid.py --preset all07 --out screenshots/grid_all07.png`
- 画像は `screenshots/` に保存。比較のため旧/新を残す。
  - wobble 個別: `python scripts/tune_wobble.py --amplitude 2.5 --frequency 0.02 --out screenshots/wobble_default.png`
  - displace 個別: `python scripts/tune_displace.py --amplitude-mm 1.5 --spatial-freq 0.02 --out screenshots/displace_default.png`
  - offset 個別: `python scripts/tune_offset.py --distance 0.2 --join round --segments 12 --out screenshots/offset_default.png`
  - repeat 個別: `python scripts/tune_repeat.py --count 3 --offset 12 0 0 --out screenshots/repeat_default.png`
  - ripple 個別: `python scripts/tune_ripple.py --amplitude 1.5 --frequency 0.03 --out screenshots/ripple_default.png`

作業の進め方（1エフェクトあたり）
1) `scripts/tune_<effect>.py` が無ければ `scripts/tune_twist.py` を雛形に作成。
2) 数パラメータを掃引し、見やすい組み合わせを探索（画像を数点保存）。
3) `effects/<name>.py` のデフォルト引数値・docstring を更新。必要なら可視化の軽微修正。
4) スナップショットを取り直し、効果の意図が明確か確認。
5) 本チェックリストの該当項目にチェックを入れ、決定値と根拠を簡記（追記歓迎）。

既に調整済み（参考）
- twist: `angle=60°`（視認性と収まりのバランス）
- fill: `mode="lines"`, `density=0.35`, `angle_rad=π/4`。dots モードはレンダラ都合で点を短い十字線に置換し可視化修正済み。
- wobble: `amplitude=2.5`, `frequency=0.02`, `phase=0.0`
  - 根拠: 300mm 正方キャンバス中央の立方体（辺=150mm）に対して、約3周期（0.02 cycles/mm ≒ 1周期/50mm）のうねりで輪郭の曲率が十分に見え、2.5mm の変位は破綻やはみ出しを避けつつ視認性が高い。
  - 再現: `python scripts/tune_wobble.py --amplitude 2.5 --frequency 0.02 --out screenshots/wobble_default.png`
  - 注意: 形状が線分少・頂点密度低のときは `subdivide` を併用（例: `--subdivisions 0.8`）。
 - explode: `factor=0.3`（約15mmの放射オフセット）
  - 根拠: `effects/explode.py` は `factor∈[0..1] → 0..50mm` に線形写像。キャンバス300mm・立方体150mmの構成で、+15mm の放射移動は効果が一目で分かりつつ余白を十分に確保（自己交差や画面外逸脱なし）。
  - 再現: `python scripts/tune_explode.py --factor 0.3 --out screenshots/explode_default.png`
 - weave: `num_candidate_lines=0.2`, `relaxation_iterations=0.3`, `step=0.25`
  - 根拠: `num_lines=100`, `iters=15`, `step≈0.125` に写像。単一立方体でウェブが十分密に形成され、計算コストと視認性のバランスが良好。
  - 再現: `python scripts/tune_weave.py --lines 0.2 --iters 0.3 --step 0.25 --out screenshots/weave_default.png`
- explode: `factor=0.3`（約15mmの放射オフセット）
  - 根拠: `effects/explode.py` は `factor∈[0..1] → 0..50mm` に線形写像。キャンバス300mm・立方体150mmの構成で、+15mm の放射移動は効果が一目で分かりつつ余白を十分に確保（自己交差や画面外逸脱なし）。
  - 再現: `python scripts/tune_explode.py --factor 0.3 --out screenshots/explode_default.png`

- dash: `dash_length=6.0mm`, `gap_length=3.0mm`
  - 根拠: 300mm 正方キャンバス中央の立方体（辺=150mm）の各エッジに対し、パターン長 9mm（6+3）で約 16〜17 ダッシュ/辺となり、過密にならず破線らしさが明瞭。
  - 再現: `python scripts/tune_dash.py --dash_length 6 --gap_length 3 --out screenshots/dash_default.png`

- ripple: `amplitude=1.5mm`, `frequency=0.03`, `phase=0.0`
  - 根拠: 150mm エッジ長に対し 0.03 cycles/mm は約 4.5 周期。1.5mm の変位で波の曲率が明確になり、過度な歪みや自己交差を避けられる。
  - 再現: `python scripts/tune_ripple.py --amplitude 1.5 --frequency 0.03 --out screenshots/ripple_default.png`
  - 注意: 頂点密度が低い線形状では `subdivide` を併用（例: `--subdivisions 0.7`）。

- displace: `amplitude_mm=1.5`, `spatial_freq=0.02`, `t_sec=0.0`（プレビュー時は `subdivisions=0.7` を推奨）
  - 根拠: 300mm キャンバス・立方体（辺=150mm）に対し、Perlin 由来のソフトな変形を視認可能にしつつ、エッジの激しい折れや画面外逸脱を避ける妥協点として 1.5mm を採用。`spatial_freq=0.02` は概ね 50mm スケールの起伏になり、線分が過密にならず形状の把握が容易。
  - 再現: `python scripts/tune_displace.py --amplitude-mm 1.5 --spatial-freq 0.02 --subdivisions 0.7 --out screenshots/displace_default.png`
  - 参考（異方性）: `python scripts/tune_displace.py --amplitude-mm 1.0 --spatial-freq 0.03 0.01 0.00 --out screenshots/displace_xyz.png`

- offset: `join='round'`, `segments_per_circle=12`, `distance=0.2`（0..1→約5mm）
  - 根拠: 300mm 正方・中心立方体に適用時、視認性の高い拡張境界が得られつつ、過度な膨張や角の破綻を回避。`segments_per_circle=12` で円滑さと描画コストのバランス。
  - 再現: `python scripts/tune_offset.py --distance 0.2 --join round --segments 12 --out screenshots/offset_default.png`

- repeat: `count=3`, `offset=(12,0,0)`, `angles_rad_step=(0,0,0)`, `scale=(1,1,1)`, `pivot=(0,0,0)`
  - 根拠: 300mm 正方・中心立方体で横並び4体（元＋3複製）が余白内に収まり、効果が明確。変換は複製番号に比例して累積するため、1ステップ12mmに設定し視認性と収まりを両立。
  - 再現: `python scripts/tune_repeat.py --count 3 --offset 12 0 0 --scale 1 1 1 --out screenshots/repeat_default.png`

注意事項 / Known gotchas
- dots: レンダラは点を描けないため、短い線分で表現（`effects/filling.py` 内で実装済み）。
- offset: Shapely 依存。未導入環境ではテスト不可。可視調整のみ、失敗時は元ジオメトリ返し。
- キャッシュ: 形状/パイプラインキャッシュが効くため、パラメータ掃引時に描画が速くなることあり。

- dash: `dash_length>0` が前提（0 は無意味）。極端に短い長さは線分過密・描画負荷増に繋がる。端部は補間により部分ダッシュになり得る。2頂点未満の線分は変更なし。

- ripple: 頂点密度不足だと各線分がほぼ平行移動に見えやすい。`subdivide` を前段に入れて曲率を確保。

- displace: `spatial_freq` を上げすぎると（目安 > 0.10） aliasing 的に細かすぎる起伏になり、線分がギザついて視認性が低下。静止スナップショット評価では `t_sec=0.0` を推奨。`__param_meta__` に `spatial_freq: number|vec3 (min>=0)` を追記する改善余地あり（別PR）。

# Effect Tuning Checklist (Not Yet Tuned)

Generated: 2025-09-06

対象: `effects/` に登録済み（@effects.registry.effect）で、まだデフォルト調整・検証を行っていないエフェクト。

- [x] affine — default: pivot=None (centroid), angles_rad=(0,0,0.35), scale=(1.1,0.85,1.0)
- [x] collapse — default: intensity=1.8, subdivisions=0.5 (強め)
- [x] dash — default: dash_length=6.0, gap_length=3.0
- [x] displace — default: amplitude_mm=1.5, spatial_freq=0.02, t_sec=0.0（プレビュー推奨: subdivisions=0.7）
- [x] explode — default: factor=0.3
- [x] extrude — default: distance=0.35, scale=0.35, subdivisions=0.3, center_mode=auto
- [x] offset — default: distance=0.2 (~5mm), join='round', segments_per_circle=12
- [x] repeat — default: count=3, offset=(12,0,0), angles_rad_step=(0,0,0), scale=(1,1,1), pivot=(0,0,0)
- [x] ripple — default: amplitude=1.5, frequency=0.03, phase=0.0
- [ ] rotate — primary: pivot, angles_rad
- [ ] scale — primary: pivot, scale
- [ ] subdivide — primary: subdivisions
- [ ] translate — primary: delta
- [x] trim — default: start_param=0.1, end_param=0.9
- [x] weave — default: num_candidate_lines=0.2, relaxation_iterations=0.3, step=0.25


備考:
- 既に詰め済み（除外）: `fill`（lines/density=0.35/angle=π/4, dots 実装修正済み）, `twist`（angle=60°）。
- 進行に合わせて本チェックリストは更新してください。
