# Geometry 改善チェックリスト（src/engine/core/geometry.py）

本チェックリストは、`Geometry` 実装の品質・一貫性・性能を高めるための具体的アクションを短く分解したもの。実装前にあなたの確認を得て、承認項目から順に適用します。進行中は本ファイルに進捗と補足を逐次追記します。

## 背景/目的

- 仕様（不変条件/純関数性）を保ったまま、ドキュメント整合と性能を改善。
- digest 方針（lazy/eager）の記述と実装のズレ解消。
- 防御的プログラミング（不変条件検証/参照の書込禁止）で安全性を向上。

## スコープ

- 対象: `src/engine/core/geometry.py`
- 付随: テスト（`tests/`）、`architecture.md`（digest ポリシー同期）

---

## 合意が必要な確認事項（要回答）

- [YES] A1: digest は「完全 lazy」に統一して良いか（変換直後の eager 計算はしない）
- [ ] A2: `as_arrays(copy=False)` は read-only ビューを返す仕様に変更して良いか（互換性注意）
- [ ] A3: `rotate` は `float32` の回転行列で一括適用に切替で良いか（数値誤差は 1e-6 以内）
- [ ] A4: 追加ユーティリティ（`num_points`, `to_lines()`, `bounding_box()`）を今回入れるか（任意）

---

## タスク一覧（完了条件つき）

### 1) Digest ポリシー整合（lazy 化）

- [ ] T1-1: 変換/生成直後に digest を計算しない実装へ変更（`_set_digest_if_enabled` 呼出の撤廃）
  - DoD: 変換後 `._digest is None`。`g.digest` 初回アクセスのみ計算・以後キャッシュ。
- [ ] T1-2: docstring/コメントを「必要時に遅延計算」に統一
  - DoD: すべての該当 doc に反映、行長 100 内に整形。
- [ ] T1-3: `PXD_DISABLE_GEOMETRY_DIGEST=1` 時の例外説明を最新化
  - DoD: 例外メッセージ/ドキュメントが一致。
- [ ] T1-4 (任意): マイクロベンチの TODO を tests にスキップ付きで記録

### 2) 不変条件の強化

- [ ] T2-1: `__post_init__` を追加し `coords/offsets` の型・形状・整合を検証
  - ルール: `coords.dtype=float32`, `coords.shape[1]==3`, `offsets.dtype=int32`, `offsets[0]==0`, 単調増加, `offsets[-1]==len(coords)`
  - DoD: 不正入力で `ValueError` を投げ、メッセージに原因を含める。

### 3) `as_arrays` の安全性

- [ ] T3-1: `copy=False` の返り値を read-only 化（`setflags(write=False)`）※A2 承認必須
  - DoD: 返却配列に対する書込で `ValueError`（Read-only）を確認。docstring に仕様明記。
- [ ] T3-2: 逆に書込を許可したい場合の回避策を doc に記述（`copy=True` を使用）

### 4) 幾何変換の最適化/一貫性

- [ ] T4-1: `translate` を `copy()+=vec` 方式へ統一（テンポラリ削減）
  - DoD: プロファイル上メモリアロケーション 1 回に減る（目視/コメント根拠で可）。
- [ ] T4-2: `rotate` を `float32` 回転行列で一括適用に変更（※A3 承認必須）
  - DoD: 旧実装との数値差が `1e-6` 以下のテストを追加。

### 5) `from_lines` の堅牢化/高速化

- [ ] T5-1: 2D→3D 補完を `empty` 事前確保で実装（`hstack` 回避）
  - DoD: 生成配列は `float32 (n,3)`、不要な一時配列なし。
- [ ] T5-2: 例外メッセージに行インデックス/shape を付与
  - DoD: `lines[i]` の何が不正か分かる文面。
- [ ] T5-3: 0 次元入力（スカラー）を明示的に `ValueError`
  - DoD: 該当ケースのテストを追加。

### 6) スタイル/ドキュメント整備

- [ ] T6-1: すべての公開 API を NumPy スタイル docstring に統一（日本語・事実記述）
- [ ] T6-2: 行長 100 を満たすよう整形
- [ ] T6-3: Lint/Format/Type（単一ファイル）を通過
  - DoD: `ruff/black/isort/mypy` が当該ファイルで成功。

### 7) 追加ユーティリティ（任意・後回し可）

- [ ] T7-1: `num_points` プロパティ
- [ ] T7-2: `to_lines()`（各ポリラインの `view`/コピーの方針は read-only ベース）
- [ ] T7-3: `bounding_box()`（min/max を返す）
- [ ] T7-4: `__eq__`（配列同値/型同値; digest 依存は避ける）

### 8) ドキュメント同期

- [ ] T8-1: `architecture.md` に digest の lazy 方針を追記（関連コード参照を明記）

### 9) テスト（代表）

- [ ] TT-1: `from_lines` 正規化（2D/3D/1D/空/0 次元エラー）
- [ ] TT-2: `rotate`（各軸/合成/ピボット）
- [ ] TT-3: `concat`（offset シフト/型維持）
- [ ] TT-4: digest（lazy 計算、環境変数無効時の例外）
- [ ] TT-5: `as_arrays(copy=False)` での read-only 動作（※A2 承認時）

---

## 実行コマンド（編集ファイル優先の高速ループ）

- Lint: `ruff check --fix src/engine/core/geometry.py`
- Format: `black src/engine/core/geometry.py && isort src/engine/core/geometry.py`
- TypeCheck: `mypy src/engine/core/geometry.py`
- Test (targeted examples):
  - `pytest -q -k geometry`
  - `pytest -q tests/test_geometry_model.py::test_from_lines_normalizes_shapes`（例）

## リスクと緩和

- `as_arrays(copy=False)` の read-only 化は互換性影響あり → A2 で方針決定。必要なら段階導入（警告ログ → 将来変更）。
- `rotate` の行列化でわずかな丸め誤差 → 許容誤差をテストで明示。
- `__post_init__` の検証で既存の不正呼出しが露見 → 失敗時は呼出し側を修正し、メッセージを具体化。

## 変更ログテンプレ（PR 用）

- feat(core/geometry): digest lazy 化、**post_init** 追加、rotate 最適化、from_lines 強化 ほか
- テスト: 上記に対応する単体テスト一式を追加
- 検証: ruff/black/isort/mypy OK、pytest スモーク OK

---

更新履歴:

- 2025-09-14: 初版作成（レビュー待ち）
