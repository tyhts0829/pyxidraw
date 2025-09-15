# AsemicGlyph 関数化計画（shape の関数ベース統一・大型対応）

日付: 2025-09-15
提案者: チーム
対象: `src/shapes/asemic_glyph.py`, `src/shapes/__init__.py`, `src/scripts/gen_g_stubs.py`, `tests/shapes/*`

## 背景
- リポは「shape を関数ベースへ全面統一」方針に移行済み（多くの shape を関数化完了）。
- `AsemicGlyph` は規模が大きく、numba / SciPy（`scipy.spatial.cKDTree`）に依存するため、個別計画として切り出す。
- 目的はクラス `AsemicGlyph(BaseShape)` を `@shape def asemic_glyph(...) -> Geometry` へ移行すること。

## 目的（Goals）
- 既存の動作/品質（ランダム生成・滑らか化・補助記号の付与）を維持したまま関数化。
- 公開 API から `BaseShape` 依存を完全に排除。
- import 時の依存コストを抑え、IDE 補完（スタブ生成）とテスト安定性を保つ。

## 非目的（Non-goals）
- 生成アルゴリズムの変更・高機能化（性能調整以外）。
- SciPy 依存の除去自体（ただし遅延 import やフォールバックを検討）。

## 設計方針
- シグネチャ: `def asemic_glyph(*, region=(-0.5, -0.5, 0.5, 0.5), smoothing_radius=0.05, diacritic_probability=0.3, diacritic_radius=0.04, random_seed=42.0, **_params) -> Geometry`。
- 変換は Geometry 側のメソッド連鎖で実施するため、本関数は「原点基準の素の形状」を返すだけに専念。
- 既存の補助関数は極力モジュールレベル関数として再利用（`AsemicGlyphConfig` は dataclass のまま）。
- 乱数は `random.Random(int(random_seed))` を関数内に閉じる（再現性）。
- SciPy の `cKDTree` は「遅延 import + フォールバック」戦略:
  - `try: from scipy.spatial import cKDTree` を関数内で実行。
  - ImportError 時は `numpy` の全対比較（O(N^2)）にフォールバック（小規模 N 前提）。
  - N が閾値を超える場合は `RuntimeError("SciPy not available for large N")` を推奨（閾値は 1,500 など）。

## 実装アクション（詳細）
1) 構造の整理（非破壊）
- [x] `asemic_glyph.py` 内の `AsemicGlyph.generate()` 本体を読み取り、依存ヘルパ（`generate_nodes`, `relative_neighborhood_graph`, `random_walk_strokes`, `snap_stroke`, `smooth_polyline`, `add_diacritic` など）と I/O（Geometry 生成）境界を明確化。
- [x] ランダム関連（`random_seed` の適用箇所）を特定し、局所乱数器に統一。

2) 関数化
- [x] `@shape def asemic_glyph(...) -> Geometry` を実装し、`generate()` のロジックを移植。
- [x] 既存のクラス依存を解消（モジュール関数に統一）。
- [x] `relative_neighborhood_graph` 内の `cKDTree` 参照を「遅延 import + フォールバック」へ改修。
- [x] 返り値は `Geometry.from_lines(vertices_list)` で統一。

3) 登録と import コストの最小化
- [x] `shapes/__init__.py` で `from . import asemic_glyph as _register_asemic_glyph` を追加（副作用登録）。
- [x] `asemic_glyph.py` 先頭の重い import（SciPy）を削除し、関数内遅延 import へ移動。
- [x] `numba`/`numpy` は既存のダミーでカバー、SciPy は遅延化で回避。

4) スタブと API 整合
- [x] `gen_g_stubs.py` は関数シグネチャから自動取得済みのため変更不要。
- [x] スタブ再生成後、`G.asemic_glyph(...)` が自動反映されることを確認。

5) テスト
- [x] 登録テスト: `is_shape_registered("asemic_glyph")`、`get_shape("asemic_glyph")` が関数を返す。
- [x] 生成スモーク: 固定 `random_seed` での出力が `Geometry` で非空である。
- [x] フォールバック動作: SciPy を無効化＋`generate_nodes` をモックし、小規模 N でフォールバックを確認。
- [ ] 例外動作: フォールバック閾値超過時の RuntimeError を確認（任意）。

6) ドキュメント/ガイド
- [ ] `docs/proposals/shape_function_only_migration.md` のチェックリストで `asemic_glyph` を [x] に更新。
- [ ] `docs/user_extensions.md` へ関数ベースの使用例に注記（重い依存の遅延 import 方針）。

7) リスク低減
- [ ] import コストの監視: `import shapes` 時に SciPy を触らないことを維持（関数内遅延 import）。
- [ ] ランダム性: `random_seed` のみが出力に影響することをテストで保証（外部乱数状態に依存しない）。

## 受け入れ基準（DoD）
- [x] `asemic_glyph` が関数として登録・利用可能（`G.asemic_glyph(...)`）。
- [x] 既存の `AsemicGlyph` クラスを削除し、`BaseShape` 参照がゼロ。
- [x] スタブ再生成で `G` に `asemic_glyph(...)` が含まれる。
- [x] 変更ファイルに対する `ruff/black/isort/mypy` が通過。
- [x] テスト緑（スモーク/フォールバック）。

## スケジュール（目安）
- 実装 + 局所テスト: 0.5–0.75 日
- フォールバック/例外テスト・微調整: 0.25–0.5 日

## 備考（トレードオフ）
- SciPy 非導入環境でも最低限動作するようフォールバックを用意するが、大規模 N の性能は劣化する。実運用では SciPy 援用を推奨（ドキュメント化）。
- numba 依存については既存のダミーがあるため、導入コストは現状維持。
