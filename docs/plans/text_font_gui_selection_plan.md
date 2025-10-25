**目的**
- `src/shapes/text.py` の `text` 形状において、フォント名（`font`）を Parameter GUI から選択できるようにする。
- 既存のルール（明示引数 > GUI > 既定値）と `__param_meta__` ベースの UI メタに準拠する。

**現状と課題**
- 現状は `font` を文字列引数として受け付けるが、GUI には出さない実装になっている。
  - 根拠: `src/shapes/text.py:400-402` で `"font": {"choices": []}` を設定（空 choices → enum だが supported=False 扱い → 非表示）。
- ユーザーは GUI 上でフォントを切り替えられず、コード/設定の編集が必要。
- `.ttc` のフェイス選択は `font_index`（int）で対応済みだが、フォント本体の選択性が不足。

**要件（機能）**
- Parameter GUI に `font` を列挙型（enum/コンボ）として表示する。
- 候補は設定のフォント検索ディレクトリ（`configs/default.yaml` の `fonts.search_dirs`）のみから構成する（OS フォントは含めない）。
- 表示名はファイル名の stem（拡張子・ディレクトリ除去）を基本とする。
- 選択結果は `text(font=<選択名>)` の実引数として反映し、既存の部分一致ロジックで解決（パス/部分一致/フォールバック）。
- 大量の候補がある場合は自動的にコンボボックス表示（DPG 側の既定しきい値 >5）。

**非要件/スコープ外**
- `.ttc` 内のフェイス名の列挙/選択（`font_index` の動的上限など）は本計画から除外（別計画あり: `docs/improvements/font_index_gui_dynamic.md`）。
- 家族名（name テーブル）での厳密な正規化・重複統合は行わない（将来拡張）。

**設計方針**
- `__param_meta__` の `font` に動的 `choices` を与える。
  - モジュール import 時に 1 回だけ列挙し、stem の重複を除去 → アルファベット昇順で安定化。
  - 列挙結果が空の場合は、従来どおり `choices: []` を維持（GUI 非表示のまま）。
  - 既定値 `"SFNS"` の強制挿入は行わない（OS フォントへ導く選択肢を避ける）。
- 候補生成は `util.fonts.resolve_search_dirs` と `util.fonts.glob_font_files` を用い、OS フォントは明示的に含めない。

**UI/UX**
- `text` を明示引数なしで呼ぶ（既定値採用）場合のみ GUI に `font` コンボが出現。
- 選択変更は即時反映（ParameterRuntime 経由で override 適用）。
- 候補数が多い場合でも検索なし（将来拡張で filter 等を検討）。

**エッジケースとフォールバック**
- 列挙結果が空（フォント検出不可）:
  - `choices: []` を維持し GUI 非表示。`TextRenderer` のフォールバック（OS 既定/探索済み先頭）で描画可能。
- `.ttc` のフェイス数に対する `font_index` 上限:
  - 現状据え置き（0..32）。必要に応じて `font_index` 動的レンジは別計画で対応。

- **実装ステップ（チェックリスト）**
- [x] 候補列挙関数の設計: `configs/default.yaml` の `fonts.search_dirs` からのみ列挙し、stem 一覧を生成（重複除去+昇順ソート）。
- [x] `src/shapes/text.py` の `__param_meta__` 構築箇所を修正し、列挙結果を `font.choices` に埋め込む（空のときは従来通り隠す）。
- [ ] 既存挙動との互換確認: 既定値 `SFNS`/Linux/Windows でのフォールバック経路に変化なし。
- [ ] 単体テストを追加（軽量・ファイル依存を避ける）:
  - [ ] `TextRenderer.get_font_path_list()` を monkeypatch して疑似リストを返し、`choices` が期待通りに生成されることを検証。
  - [ ] `choices==[]` のケースで GUI 非表示（Descriptor.supported=False）になることを検証。
- [ ] ドキュメント更新:
  - [ ] `docs/shapes.md` に「font は GUI から選択可能。候補は設定/OS のフォント列挙に基づく」旨を追記。
  - [ ] 必要なら `architecture.md` の該当箇所に反映（UI メタの動的要素）。

**テスト計画（実行コマンド）**
- 変更ファイル単位の高速ループ（実施済み: ruff/black/isort/mypy）:
  - Lint: `ruff check --fix src/shapes/text.py`
  - Format: `black src/shapes/text.py && isort src/shapes/text.py`
  - TypeCheck: `mypy src/shapes/text.py`
  - Unit/GUI最小: `pytest -q tests/ui/parameters -k text`（追加テストがあれば個別指定）

**リスクと緩和**
- 候補数が非常に多い環境での UI レスポンス低下 → 初回 import 時のみ列挙、DPG はコンボ描画を選択。必要時は上限/フィルタを後続で導入。
- 端末依存のフォント名の揺れ → 部分一致解決（stem）を利用し、実ファイル名に追従。

**確認事項（決定）**
- 候補に OS フォントは含めない（search_dirs のみ）。
- 候補表示は「ファイル名の stem（例: `DejaVuSans`）」で進める。
- 候補の上限は当面なし（必要なら後続で追加）。

**参考**
- 検索/列挙: `util.fonts.glob_font_files`, `util.fonts.os_font_dirs`, `util.fonts.resolve_search_dirs`
- 解決/フォールバック: `src/shapes/text.py` の `TextRenderer.get_font()`
- 既存の `.ttc` レンジ動的化の設計: `docs/improvements/font_index_gui_dynamic.md`
