# subdivide エフェクト改善計画（提案）

- 対象: `src/effects/subdivide.py`
- 目的: 可読性と堅牢性を維持しつつ、停止条件・型一貫性・軽微な性能改善を行う。
- 方針: 破壊的変更は許容（本リポは未配布）。美しくシンプルを優先。
- このチェックリストは改善の前段。承認後に実装・検証へ進む。

## 変更概要（What）

- `subdivisions` を整数パラメータとして扱い、丸め処理を廃止。
- 中止条件の強化: 先頭セグメントのみでなく「全セグメントの最小長」を判定（平方長で比較）。
- 配列確保の微最適化: `np.zeros` → `np.empty`（全要素を明示代入するため）。
- 定数の集約: `MAX_SUBDIVISIONS`, `MIN_SEG_LEN` をモジュール定数化。
- docstring を NumPy スタイルに整理（Parameters/Returns/Notes）。
- （任意）頂点爆発ガード: 想定最終頂点数の上限で早期打ち切り/クランプを検討。

## ねらい（Why）

- 型の一貫性（UI/RangeHint と一致）と API 明確化。
- 極短セグメントの存在で暴走/無駄分割を抑制。
- 軽微な割り当て削減と sqrt 回避での数値計算コスト削減。

## 実装チェックリスト（How）

- [x] モジュール定数を追加: `MAX_SUBDIVISIONS = 10`, `MIN_SEG_LEN = 0.01`（`MIN_SEG_LEN_SQ` 併用）。
- [x] 関数シグネチャを整数へ: `def subdivide(g: Geometry, *, subdivisions: int = 5) -> Geometry`。
- [x] クランプロジックを `MAX_SUBDIVISIONS` に統一（丸め/float 入力の廃止）。
- [x] `_subdivide_core` の最小長チェックを「全セグメントの二乗距離の最小値」で判定。
- [x] `_subdivide_core` の配列確保を `np.empty((2*n-1, 3), dtype=...)` に変更。
- [x] ループ内の早期停止条件も二乗距離ベースに更新（再判定）。
- [x] 公開 docstring を NumPy スタイルで整理（Parameters/Returns/Notes）。
- [x] `__param_meta__` は現状維持（0–10 の RangeHint）。
- [x] 合計頂点数の上限ガード（`MAX_TOTAL_VERTICES = 10_000_000`）を追加。

## 追加検討（要確認）

- [x] 合計頂点数上限（10,000,000 点）でガード（実装済）。
- [ ] Numba 依存の optional 化（`dash` のように環境変数で切替）を行うか。既存エフェクトとの方針統一。
- [ ] `MIN_SEG_LEN` の既定値 0.01 は妥当か（単位: 座標系 mm 相当）。UI に露出しない定数のままで良いか。
- [ ] `subdivisions` の GUI 範囲は 0–10 のままで良いか。

## 影響範囲

- 実装ファイル: `src/effects/subdivide.py`
- 公開 API: パラメータ型が `float` → `int`（破壊的。ただし UI は整数スライダーで整合）。
- 他所への副作用: なし（`Geometry.from_lines` に依存するのみ）。

## 受け入れ基準（Acceptance Criteria）

- 分割 0 回は入力コピーを返す（内容一致）。
- 2 点線: 1 回で 3 点、2 回で 5 点に増加。
- 大きすぎる分割指定は `MAX_SUBDIVISIONS` にクランプされる。
- 極短線（`MIN_SEG_LEN` 未満）では分割しない。
- 複数ポリライン入力で `offsets` が正しく再構築される。
- dtype と不変条件（`(N,3) float32` / `int32` offsets）を維持。

## 検証（編集ファイル優先の高速ループ）

- Lint: `ruff check --fix src/effects/subdivide.py`
- Format: `black src/effects/subdivide.py && isort src/effects/subdivide.py`
- TypeCheck: `mypy src/effects/subdivide.py`
- Test（追加後）: `pytest -q tests/effects/test_subdivide.py`
- 既存スモーク: `pytest -q -m smoke`（副作用確認）

## テスト計画（新規テスト案）

- [ ] 基本: 2 点線 +1 回 → 3 点、+2 回 → 5 点。
- [ ] クランプ: 100 回指定 → `MAX_SUBDIVISIONS` に縮減。
- [ ] 最小長: `MIN_SEG_LEN` 未満の線で分割されない。
- [ ] 複数ポリライン: `Geometry.from_lines` の offsets 整合。
- [ ] 不変条件: dtype/shape/offsets の妥当性検証。

## リスク / ロールバック

- API 破壊（float→int）。GUI は整数想定のため実害は小。必要なら thin 互換層（float 受理 →int 変換）を一時的に残せるが、シンプルさ優先で削除予定。
- 早期停止の強化により、従来より分割が早く止まるケースが出る可能性。`MIN_SEG_LEN` を調整可能にする方針もあり得る。
- ロールバックは当該コミットのみ差し戻し可能（単一ファイル・局所変更）。

---

承認いただければ、上記チェックリストに従って実装を進め、完了項目へチェックを入れていきます。追加の要望や閾値・方針の確定があればコメントください。
