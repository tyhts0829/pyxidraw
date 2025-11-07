# LazyGeometry: デバッグカウンタ整理と例外縮小 — 実施チェックリスト（提案）

目的
- `src/engine/core/lazy_geometry.py` に残る未使用のデバッグ要素を整理し、例外捕捉を必要最小限に縮小して、読みやすさと保守性を高める。

スコープ
- 対象: `src/engine/core/lazy_geometry.py`
- 付随: 計画ドキュメントの該当チェック項目更新（`docs/plan_refactor_lazy_cache_multiproc.md`）

前提観測（現状）
- 未使用要素
  - `_PREFIX_TAIL_SKIPPED`（定義のみ、参照なし）
  - `_PREFIX_STORE_TAIL`, `_PREFIX_STORE_ON_MISS_UP_TO`（環境変数の受け口のみ、参照なし）
- カウンタ類（`_PREFIX_HITS/MISSES/STORES/EVICTS`）は更新のみで外部から参照されていない。
- `realize()` 内に広域 `try/except Exception: pass` が複数あり、意図が読みにくい。

やること（タスク分解）
1) 未使用デバッグ要素の整理（安全・最小）
   - [x] `_PREFIX_TAIL_SKIPPED` の削除（定義・増分関数が無いため完全削除）
   - [x] `_PREFIX_STORE_TAIL`, `_PREFIX_STORE_ON_MISS_UP_TO` の削除（未使用の環境スイッチ）
   - 備考: `docs/plans/prefix_effect_cache_tail_guard.md` は将来案のため、今回はコードのみ整理（ドキュメントは据え置き）

2) 例外捕捉の縮小（必要箇所に限定）
   - [x] Prefix 検索ブロックの外側 `try/except` を撤去
   - [x] 効果適用＋保存ブロックの外側 `try/except` を撤去
   - [x] 残す `try/except` は「署名生成（`params_signature`）失敗時は“非キャッシュ”にフォールバック」だけに限定し、理由コメントを付記
   - [x] LRU 操作（`get/pop/popitem/len`）には例外捕捉を付けない（不変条件に依存）

3) デバッグカウンタの軽量化（維持する場合）
   - [x] `_PREFIX_*_INC()` の中で `_PREFIX_DEBUG` が False のときはノーオペにする（実行コスト削減）
   - [x] カウンタの用途コメントを追加（将来の HUD/ログ拡張用の計測）

4) ドキュメント更新
   - [x] `docs/plan_refactor_lazy_cache_multiproc.md` の該当 2 チェック項目を完了に更新（簡潔な注記を追記）

検証（編集ファイル限定）
- `ruff check --fix src/engine/core/lazy_geometry.py`
- `black src/engine/core/lazy_geometry.py && isort src/engine/core/lazy_geometry.py`
- 重点テスト（例）
  - `pytest -q tests/api/test_pipeline_cache.py`

質問（ご確認ください）
1) `_PREFIX_TAIL_SKIPPED` と `_PREFIX_STORE_*` の完全削除で問題ありませんか？（現状未使用）
2) カウンタ `_PREFIX_HITS/MISSES/STORES/EVICTS` は保持し、`_PREFIX_DEBUG` 偽のときノーオペ化で良いですか？（将来 HUD/ログ拡張の余地確保）
3) 例外縮小により、予期しない実装エラーはそのままテストで発覚する方針で問題ありませんか？

承認後、上記チェックリストに沿って実装します。
