# displace: spatial_freq 変更時の位相ジャンプ抑制 改善計画

## 概要
- 現状、`freq`（`spatial_freq`）を変更するとノイズ空間の位置（位相）が急峻に変化する。
- 原因は「平行移動（時間オフセット）→ スケーリング（freq）」の順で座標変換しているため、位相が `offset * freq` に依存してしまう点。
  - 現行実装参照: `src/effects/displace.py:135-139`（スケーリング）と `src/effects/displace.py:164-171`（時間オフセットの適用）
- 目標は「freq を変えるとスケールだけが変わり、位相は独立（連続）」にすること。

## 目標（Do）
- ドメイン変換を `noise(pos * freq + phase)` に統一（phase は freq に非依存）。
- `t_sec` は位相（時間）として freq 非依存で効くようにする。
- 既存 API（`displace(...)` 引数）は不変。内部実装のみ調整。

## 非目標（Don’t）
- `wobble` 等の他エフェクトへ波及変更はしない。
- 依存追加・大規模最適化は行わない。

## 変更方針（設計）
- 変換順序を「スケーリング→平行移動」に変更。
  - 具体的には、Perlin 入力を `x = pos_x * fx + phase_x`（y/z も同様）とする。
  - 現行の `offset_coords = coords + t_offset` は廃止し、`t_offset` を `phase` としてスケール後に加える。
- 位相（時間オフセット）は周波数に依存させない。
  - `phase_base = time * PHASE_SPEED + SEED` とし、`PHASE_SPEED` と `SEED` は定数。既存の体感に近づけるなら `PHASE_SPEED ≈ 10.0`、`SEED ≈ 1000.0` を踏襲。
- 軸オフセット（チャンネル分離）は現行の `+100` / `+200` を維持（freq 非依存のため問題なし）。
- 互換性: 出力の“見え方”は周波数スイープ時に連続になる（破壊的だが改善目的に適合）。

## 実装詳細（最小変更）
1) `perlin_core` に位相引数を追加（freq 後に加える）
   - 旧: `perlin_core(vertices, frequency, perm_table, grad3_array)`
   - 新: `perlin_core(vertices, frequency, phase, perm_table, grad3_array)`
   - 本体:
     - `x = vertices[i,0] * fx + phase_x`
     - `y = vertices[i,1] * fy + phase_y`
     - `z = vertices[i,2] * fz + phase_z`
     - 以降は現行の `perlin_noise_3d(...)` を踏襲（`+100`, `+200` は存置）。
2) `_apply_noise_to_coords` の前段オフセット加算を削除
   - `offset_coords = coords + t_offset` を削除。
   - `phase0 = float(time) * PHASE_SPEED + SEED`、`phase_tuple = (phase0, phase0, phase0)` を算出し、`perlin_core(..., phase_tuple, ...)` に渡す。
3) 既存 API/docstring/param_meta の説明更新（挙動の簡潔な説明に修正）。

オプション（必要なら別コミット）
- アンカー中心化: `noise((pos - center) * freq + phase)`（`center` は重心）。周波数変更時の“縮尺だけ”感がさらに自然。初期は無効で良い。
- `__param_meta__['spatial_freq']` に `step` を付与（量子化の視覚段差を抑える）。

## テスト・検証
- 単体: 小ジオメトリ（例: 正方格子）で `freq` を微小スイープし、位相ジャンプが消えているかを確認。
  - 指標例: `freq` と `freq + ε` の出力の相関/ノルム差が滑らかに変化すること。
- 時間: `t_sec` を変化させたとき、`freq` が一定なら同じ速度で“流れる”ことを目視確認。
- 性能: Numba `njit` 維持のため、引数型と配列 dtype（`float32`）を踏襲。

推奨コマンド（編集ファイル優先）
- Lint: `ruff check --fix src/effects/displace.py`
- Format: `black src/effects/displace.py && isort src/effects/displace.py`
- Type: `mypy src/effects/displace.py`
- Test（目視 or 局所）: `pytest -q -k displace`

## 作業手順チェックリスト
- [x] `perlin_core` に `phase` 引数を追加し、スケール後に加算
- [x] `_apply_noise_to_coords` の前段オフセット加算を削除
- [x] `phase0`（`t_sec` 由来）の導入と `perlin_core` への受け渡し
- [x] ドキュストリングとコメント更新（変換順の明記）
- [x] 既存のチャンネル分離（+100/+200）維持の確認
- [x] 変更ファイルに限定した `ruff/black/isort/mypy` 緑
- [ ] 動作確認（freq スイープの連続性、t_sec の独立性）

オプション（別途）
- [ ] アンカー中心化の可否フラグ追加（既定 OFF）
- [ ] `__param_meta__['spatial_freq']` に `step` 追加の検討

## 要確認事項（質問）
1) `t_sec` は「freq に依存しない位相（時間）」として定義してよいか。
2) アンカー中心化（重心基準）は導入したいか（既定 OFF で用意）。
3) `PHASE_SPEED` と `SEED` の初期値は現行の体感に合わせて `10.0` と `1000.0` で良いか。
4) GUI 側の `spatial_freq` の `step`（量子化刻み）は付けるべきか（例: `1e-3`）。

## ロールバック方針
- 旧挙動へ戻す場合は、`perlin_core` の `phase` 追加を戻し、`_apply_noise_to_coords` の前段オフセット加算を復活させる（差分は局所的）。

---
この計画で実装を進めてよいかご確認ください。修正は `src/effects/displace.py` のみ（最小差分）を想定しています。
