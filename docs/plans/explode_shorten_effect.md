# explode: 「分断して短くする」仕様変更 計画チェックリスト

対象: `src/effects/explode.py`
目的: 外側へ押し出さず、すべての連続線を線分単位に分断し、各線分を短くする。効果としては「いったん外側へ押し出し（仮想）、その後に全体を元サイズへスケールバック」した結果と同等になることを狙う。

---

## 仕様（最終）
- 入力: `Geometry`（ポリライン集合）。
- 出力: 各線分を独立ポリラインに分断。単一点ポリラインは 1 点のまま。
- 変形:
  - 「仮想」explode: 線分中点 `mid` と重心 `center` の方向へ `factor` [mm] だけ平行移動するベクトル `delta` を定義。
  - ただし実際には押し出さない。最終結果は次で直接計算する。
  - 全頂点の最大半径（`||p-center||`）を `R_before`、仮想押し出し後の最大半径を `R_after` とし、スケール係数 `s = R_before / R_after`（安全化あり）を算出。
  - 各端点は `p_final = center + s * (p + delta - center)` で直接求める（線分長は `s` 倍になり短くなる）。
  - Z も含め XYZ 方向で処理（3D 対応）。
- パラメータ: `factor: float`（RangeHint 0–50、単位 mm）。`factor=0` は分断のみで長さ不変。
- 退避/例外: `R_before` または `R_after` が極小の場合は `s=1` とする。

---

## 実装手順
- `center = coords.mean(axis=0)` を算出。
- 1 パス目（サイズ測定）:
  - `R_before = max(||coords - center||)` を計算。
  - 各線分/単点に対して `delta` を計算し、仮想後端点 `p+delta` の半径最大値から `R_after` を得る。
- スケール係数: `s = (R_before > eps and R_after > eps) ? R_before / R_after : 1.0`。
- 2 パス目（出力生成）:
  - 分断後の `out_coords/out_offsets` を事前確保。
  - 各線分/単点で `p_final = center + s * (p + delta - center)` を書き込む。
- ドキュストリング更新（「外側へ押し出さずに短縮」「分断」明記）。
- `__param_meta__`（RangeHint）は現状維持。

---

## チェックリスト
- [x] 仕様の合意（本MDの仕様でOKか）
- [x] 実装: `explode` のロジックを短縮仕様に更新
- [x] ドキュストリング更新（`src/effects/explode.py`）
- [x] 局所 Lint/Format/Type
      - `ruff check --fix src/effects/explode.py`
      - `black src/effects/explode.py && isort src/effects/explode.py`
      - `mypy src/effects/explode.py`
- [x] スモーク: `pytest -q -m smoke`
- [x] 公開API影響の確認（なし→スタブ更新不要）

---

## 想定される差異・注意点
- 「元サイズ」定義は「最大半径（全頂点）」基準とする。押し出し方向は線分中点基準。
- 形状によっては厳密な「押し出し→スケールバック」と完全一致しないケースがありうるが、視覚的には同等の短縮効果を狙う。
- 退避条件により、極端に退化した形状（全点が重心に一致）では分断のみとなる。

---

この計画で問題なければ実装に進みます。修正や別基準（例: バウンディングボックス径）をご希望ならお知らせください。
