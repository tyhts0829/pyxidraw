# effect: affine 平行移動追加 設計・実装計画

本ドキュメントは、`src/effects/affine.py` に平行移動（translate）を統合する改善計画。実装前の合意形成と、進行時のチェックリストとして使用する。

現状把握:

- 現在の `affine` は「スケール → 回転（Rz·Ry·Rx）」のみを適用し、中心は `auto_center` もしくは `pivot` で決定する。
- ドキュメントに「平行移動は別エフェクト（translate）」と明記されている（参照: `src/effects/affine.py:13`).
- Pipeline スタブでは `affine(auto_center, pivot, angles_rad, scale)` のみ（参照: `src/api/__init__.pyi:52`).
- 別途 `effects/translate.py` が存在し、`delta: Vec3` を受ける。

## 目的と方針

- 目的: 一般的な「アフィン合成」を 1 ステップで完結させ、パイプライン組成の簡素化と UI 一貫性を向上。
- 適用順序（v0, 確定案）: pivot 中心で「スケール → 回転」を適用した後、ワールド座標系で平行移動を加算する。
  - 数式イメージ: `p' = (R · (S · (p - c))) + c + t`（`c`: center, `t`: 平行移動 `delta`）。
  - 互換性: 既定 `delta=(0,0,0)` とし、現行の挙動は不変。

## API / パラメータ（提案）

- シグネチャ拡張: `affine(g, *, auto_center=True, pivot=(0,0,0), angles_rad=(...), scale=(...), delta=(0,0,0))`
  - 追加: `delta: Vec3` 平行移動量 [mm]。
  - 名前は `effects.translate` に合わせて `delta` を採用（要確認）。
- `__param_meta__` 追記案:
  - `delta: { type: "vec3", min: (-500, -500, -500), max: (500, 500, 500) }`
    - 範囲は `effects.translate` に合わせ ±500 を第一候補（要確認）。
    - 量子化 `step`: 未指定の既定 1e-6 を採用（または UI 操作性優先で 0.1 を指定、要確認）。

## 実装詳細

- 変換カーネル `_apply_combined_transform(...)` に `translate` ベクトル（`np.ndarray(shape=(3,), float32)`）引数を追加。
  - 実装: `transformed = rotated + center + translate` へ拡張。
  - Numba `@njit` のシグネチャ更新に伴い、呼び出し側の `np.array(delta, dtype=np.float32)` を渡す。
- 早期リターン条件の更新:
  - 旧: scale=1 かつ angles≈0 のとき恒等 → コピー返却。
  - 新: 上記に加えて `delta==(0,0,0)` のときのみ恒等。`delta` 非ゼロなら必ず実行。
- docstring 更新:
  - 「平行移動は別エフェクト」を削除し、合成順序と意味を明記。
- `__param_meta__` 更新:
  - `delta` エントリを追加（範囲・step は上記）。

## ビルド/テスト手順（編集ファイル限定）

- Lint: `ruff check --fix src/effects/affine.py`
- Format: `black src/effects/affine.py && isort src/effects/affine.py`
- TypeCheck: `mypy src/effects/affine.py`
- Tests（個別）:
  - `pytest -q tests/effects/test_affine_set_center.py`
  - 追加予定: `pytest -q tests/effects/test_affine_translate.py`
- スタブ再生成（要承認）:
  - `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

## テスト計画（追加）

- 新規 `tests/effects/test_affine_translate.py`：
  - 平行移動のみ（scale=1, angles=0, delta≠0）で期待通り `coords + delta` になる。
  - 回転+スケール後に delta が「後置加算」として作用すること（単点または正方形で検証）。
  - `auto_center` と `pivot` の切替は平行移動結果に影響しない（回転・スケール結果差分のみ）。
  - 空ジオメトリ入力の恒等処理を保持。
- 既存テストの回帰確認:
  - `tests/effects/test_affine_set_center.py` がグリーンのまま（デフォルト delta=0.0）。
  - `tests/smoke/test_fill_rotation_invariance.py` の結果不変。

## ドキュメント/同期

- `src/effects/affine.py:6` 以降の docstring を更新（合成順序、delta 追加）。
- `architecture.md`：`affine` の説明に平行移動の統合を反映（必要箇所の最小追記）。
- 公開 API スタブ `src/api/__init__.pyi` を再生成し、`affine(..., delta=...)` を反映。

## 性能・互換性

- 計算量増は加算 1 回のみで無視可能。Numba カーネル内で完結させる。
- 既定値での互換性は維持（既存パイプラインは影響なし）。

## 要確認事項（Please confirm）

- パラメータ名: `delta`（effects.translate と合わせる）でよいか。`translate`/`offset` の代替も可。：delta でおｋ
- UI Range: `delta` の範囲は ±500（translate に合わせる）か、±300（pivot に合わせる）か。；500 で
- 量子化 step: 既定 1e-6 のままか、UI 操作性優先で 0.1 を付与するか。；既定のままで
- `effects.translate` の並存方針: 当面は併存で問題ないか（deprecate は別計画）。；はい

## 作業チェックリスト（進行管理）

- [x] 実装インタフェース承認（本計画の「要確認事項」確定）
- [x] 変換カーネルに `translate` を追加（Numba シグネチャ更新）
- [x] 早期リターン条件を `delta` 対応に更新
- [x] `affine` 本体へ `delta` 引数を追加し配線
- [x] docstring 更新（合成順序/注意/既定値）
- [x] `__param_meta__` に `delta` を追加
- [x] 追加テスト `tests/effects/test_affine_translate.py` 作成
- [x] 変更箇所の ruff/black/isort/mypy を通す
- [x] 既存テストの回帰確認（対象限定で実行）
- [ ] スタブ再生成と同期テスト緑化（要承認）
- [x] `architecture.md` の最小追記（差分と参照箇所を明記）

---

更新履歴:

- v0（初版）: 設計・手順・確認事項を起案（作成者: agent）。
