# Geometry 改善チェックリスト（src/engine/core/geometry.py）

作成日: 2025-09-14  
目的: 「単一の Geometry 表現」の堅牢性・性能・可読性・ドキュメント整合を高める。コード変更はこのチェックリストの合意後に着手する。

## 対象範囲

- 実装: `src/engine/core/geometry.py`
- 付随ドキュメント: `architecture.md`（実装差分の反映）
- テスト: `tests/`（Geometry 関連の追加/更新）
- API スタブ: `src/api/__init__.pyi`（必要時のみ再生成）

## 決定事項（2025-09-14 合意済み）

- [x] 等価比較の方針: `@dataclass(eq=False)` にする
- [x] `digest` の方針: Lazy（初回 `digest` アクセス時に計算）に統一
- [x] `digest` 取得 API: 無効化時は `RuntimeError` を維持（`maybe_digest()` は追加しない）
- [x] `offsets` のコピー戦略: 常にコピー（現状維持）
- [x] 公開属性の私有化: 変更なし（現状維持）

## タスク一覧（チェックリスト）

- [x] 等価比較の安全化（`@dataclass(eq=False)`）
  - 対応: dataclass の自動 `__eq__` 生成を無効化。
  - DoD: 配列の要素比較による例外や意図しない真偽値計算が起きないこと。
  - 影響: `src/engine/core/geometry.py`, `tests/test_geometry_model.py`（新規）
<!-- 見送り: ご要望によりストリクト検証モードは実装対象外 -->
- [x] `rotate` の dtype 最適化
  - 対応: `cos/sin` の係数を `float32` 化して一時配列の `float64` 化を抑制。
  - DoD: 既存テストの数値一致（許容誤差内）＋軽量ベンチで回帰なし。
- [x] `digest` 仕様の実装/記述の統一（Lazy へ）
  - 対応: 生成・変換直後の即時計算を撤去し、`digest` プロパティ初回アクセス時に計算・保持するよう変更。docstring を Lazy に合わせて更新。
  - DoD: 実装と doc の齟齬ゼロ。ストレステストで再計算/キャッシュが期待通り。
- [ ] 読み取り専用ビューの運用整理
  - 対応: `as_arrays(copy=False)` の仕様を明文化し、外部からの就地変更ができないことをテストで保証。
  - DoD: テストで `setflags(write=False)` を確認し、不変条件を維持。
- [x] docstring の NumPy スタイル統一（日本語）
  - 対応: `from_lines`, `as_arrays`, `translate/scale/rotate/concat`, `digest` を更新。
  - DoD: Parameters/Returns/Raises/Notes の体裁に統一。
- [ ] 例示の追加（`rotate` の適用順 X→Y→Z）
  - 対応: 1 点の 90° 回転例を Notes に追加。
  - DoD: 誤解防止の最小例があること。
- [ ] `from_lines` の強化（必要なら）
  - 対応: 超多数の小ポリラインでの前確保方式の検討（現状で十分ならスキップ）。
  - DoD: ベンチで優位性がある場合のみ採用。
- [x] `architecture.md` 同期（確認）
  - 対応: digest 方針（Lazy）/不変条件の記載を確認。差分なしのため変更不要。
  - DoD: 差分レビューで実装と完全一致。
- [ ] テストの追加
  - 正規化: 2D/3D/1D/空/混在の `from_lines`。
  - 変換: 空ジオメトリの no-op、組み合わせ（translate→scale→rotate）。
  - 回転: 単位ベクトルの既知回転（90°）。
  - ダイジェスト: 有効/無効時、変更での変化検知。
  - 等価比較/ビュー: 比較が決定的、ビューが読み取り専用。
  - [x] 追加済み: `tests/core/test_geometry_more.py`, `tests/core/test_geometry_as_arrays.py`
  - [x] `as_arrays(copy=False)` の読み取り専用性とメモリ共有のテスト（`tests/core/test_geometry_as_arrays.py`）。

## 影響範囲

- コード: `src/engine/core/geometry.py`
- テスト: `tests/test_geometry_model.py`（新設）ほか必要に応じ追加
- ドキュメント: `architecture.md`、本チェックリスト（本ファイル）
- スタブ: 公開 API 変更時のみ `python -m scripts.gen_g_stubs` 実行

## 実施順序（小さな PR 方針）

1. 等価比較の安全化＋最小テスト
2. rotate dtype 最適化（非機能変更）
3. digest 方針の統一（実装 or doc）、関連テスト
4. ドキュメント整備（docstring/architecture.md）
5. 必要なら from_lines 最適化

## 検証コマンド（変更ファイル優先）

- Lint: `ruff check --fix {changed_paths}`
- Format: `black {changed_paths} && isort {changed_paths}`
- Type: `mypy {changed_paths}`
- Test: `pytest -q -k geometry` または対象ファイル指定（例: `pytest -q tests/test_geometry_model.py`）
- スタブ（必要時）: `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`

## リスクとロールバック

- 公開属性の私有化は潜在的ブレイキング変更。採用する場合は段階移行（Deprecation ノート＋ 1 リリース猶予）を推奨。
- digest 方針変更はキャッシュヒット率や初回遅延に影響。軽量ベンチで確認してから導入。

---

更新履歴:

- 2025-09-14: 初版作成（コード変更なし）
