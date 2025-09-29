# src/shapes/text.py 再設計計画（破壊的変更 OK・クリーン/効率優先）

ビジョン

- シンプルで明確な API。最小限の依存で高速・安定。
- 実装は小さく、責務が明快（フォント解決/レイアウト/アウトライン生成）。
- 使い勝手重視（直感的なサイズ/整列/フォント指定）。

スコープ

- 対象: `src/shapes/text.py` 全面整理（必要に応じ `architecture.md` 更新）。
- 破壊的変更を許容（既存パラメータ名・既定値の変更、未使用コード削除）。

---

破壊的変更（案）

- パラメータ名の刷新:
  - `font_size` → `em_size_mm`（1em の物理サイズ[mm]）。
  - `font_number` → `font_index`（TTC のサブフォント番号）。
  - `align` → `text_align`（`left|center|right`）。
- 既定値変更:
  - `em_size_mm` 既定 10.0（従来 0.4 の正規化値を廃止）。
- 未使用/冗長コードの削除:
  - 未使用の Numba 関数群（バッチ/コマンド変換高速版）。
  - 迷いのある座標センタリング `+0.5` を廃止し、ベースライン基準に統一。

新 API 仕様（shape: `text`）

- 引数（最小）:
  - `text: str` 文字列（改行 `\n` 可）。
  - `em_size_mm: float = 10.0` 1em の高さ[mm]。
  - `font: str = "Helvetica"` フォント名 or パス。
  - `font_index: int = 0` `.ttc` 向けサブフォント番号。
  - `text_align: str = "left"` 行単位の整列。
  - `tracking_em: float = 0.0` 文字間隔[em] 追加分（単純なトラッキング）。
  - `line_height: float = 1.2` 行間倍率（em 基準）。
  - `flatten_tol_em: float = 0.01` 平坦化許容差（em 基準）。
- 返り値: `Geometry`（mm 単位）。
- メタ: `__param_meta__` は実値レンジに整合（数値は min/max/step を設定、列挙は choices）。

内部設計（簡潔）

1. フォント解決
   - OS 別フォントディレクトリを列挙。`font` がパスなら優先採用。
   - `.ttc` は `font_index` を int 強制。範囲外は 0。
   - 既定フォントは OS 別に選択（macOS: Helvetica, Linux: DejaVu Sans, Windows: Arial）。見つからなければ探索リスト先頭。
2. レイアウト
   - 文字ごとの advance を合計。トラッキングは `tracking_em * unitsPerEm` を加算。
   - 改行で複数行に分割、`text_align` に応じて x オフセットを算出。
   - カーニング/複雑書記は対象外（単純な advance のみ）。
3. アウトライン → ポリライン
   - `RecordingPen → FlattenPen`。`flatten_tol_em` から `approximateSegmentLength` を導出。
   - 頂点は EM 正規化 → 物理 mm へ変換（`em_size_mm`）。Y 軸方向は上向きを正に統一。
4. キャッシュ
   - フォント/グリフコマンドの LRU（サイズ上限付き）。np.float32 で軽量化。

実装ステップ

1. 即時修正（落ちる原因の排除）
   - [ ] `.ttc` `font_index` を関数入口で `int()` 強制、負値は 0。
   - [ ] 既定フォントパスの存在確認 + フォールバック。
2. API 刷新/ドキュメント
   - [ ] 新パラメータへ切替（上記「新 API 仕様」）。
   - [ ] 先頭ヘッダ docstring と `text()` の NumPy スタイル docstring 追加。
   - [ ] `__param_meta__` を実値レンジに整合。
3. 実装整理/単純化
   - [ ] ベースライン基準へ統一（`+0.5` 中心化ロジックを削除）。
   - [ ] 未使用の Numba 関数を削除し、ホットパスのみ残す。
   - [ ] フラットニング許容差を em 基準へ変更し、導出を一本化。
4. 可搬性/探索強化
   - [ ] OS 別フォントディレクトリの拡充と順序定義。
   - [ ] 既定フォントの選定ロジック整備（明示ログ）。
5. キャッシュ/性能
   - [ ] `_glyph_cache` を LRU（上限例: 4096 glyphs）。
   - [ ] 頂点 dtype を float32 化。
6. 検証/整合
   - [ ] 変更ファイルに限定して `ruff/black/isort/mypy` を実行。
   - [ ] `texts.py` でスモーク（`.ttc`/`.ttf` 両系統）。
   - [ ] `architecture.md` に差分反映（該当コード参照行付き）。

受け入れ条件（DoD）

- `.ttc`/`.ttf` いずれでも `G.text(...)` が実行でき、`texts.py` が落ちない。
- `font_index` を float で渡しても問題化しない（int 強制）。
- `__param_meta__` と実装の意味が一致（`em_size_mm`/`flatten_tol_em` など）。
- 変更ファイルに対する `ruff/mypy/pytest -q -m smoke` が緑。

既知の非対応（意図）

- カーニング/複雑書記/縦書きは未対応。必要時は別 ADR で検討。
- HarfBuzz 等の外部依存は追加しない（Ask-first）。

オープン事項（任意回答）

- `em_size_mm` 既定値: 10.0 でよいか（調整希望があれば値）。
- `flatten_tol_em` 既定/レンジ: 0.01（0.001..0.1）でよいか。
- 既定フォントの優先順: OS ごとの希望があれば指定。

備考

- 本計画は破壊的変更を含むため、`texts.py` など呼び出し側の軽微な修正（パラメータ名変更）が発生します。
- まずは Phase 1 ～ 2 を最小で確実に入れ、その後の最適化は追って段階導入します。
