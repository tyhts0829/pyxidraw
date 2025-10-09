# run: `line_color`/`background` のヘックスカラー対応（計画とチェックリスト）

目的
- `run_sketch` の `line_color` および `background` 引数で、`#RRGGBB` / `#RRGGBBAA`（および `0x`/接頭辞なしも許容）のヘックス表記を受け付ける。
- 既存のタプル指定（RGBA/RGB）も後方互換として維持。

仕様
- 受理形式（大文字小文字不問）
  - `"#RRGGBB"`, `"#RRGGBBAA"`
  - `"0xRRGGBB"`, `"0xRRGGBBAA"`
  - `"RRGGBB"`, `"RRGGBBAA"`
- 省略時 α は 1.0（不透明）。
- 正規化: 0–255 を 0.0–1.0 へ変換。
- 不正文字列は `ValueError` を送出（明確に早期失敗）。
- タプル入力（RGB/RGBA）は従来通り 0–1 クランプして受理。

設計
- `src/api/sketch.py` 内にヘルパ `_parse_hex_color_str(s) -> tuple[float, float, float, float]` と `_normalize_color_param(value)` を実装し、`line_color`/`background` に適用。
- `RenderWindow` には正規化済 RGBA を渡し、`LineRenderer` には `line_color` を RGBA で渡す。
- `architecture.md` の呼び出し例/要点に「ヘックス文字列可」を追記。

実施チェックリスト
- [x] API: `run_sketch` の引数型に `str` を許容し docstring を更新
- [x] 実装: ヘックス解析ヘルパと 0–1 クランプを追加
- [x] 背景色: `RenderWindow(..., bg_color=...)` に正規化済 RGBA を渡す
- [x] 線色: `LineRenderer(..., line_color=...)` に正規化済 RGBA を渡す
- [x] ドキュメント: `architecture.md` にヘックス受理を明記
- [x] 検証: 変更ファイルに限定して ruff/black/isort/mypy を通す
- [ ] テスト: スタブ同期 `tests/stubs/test_g_stub_sync.py` と `-m smoke` を再実行

備考
- 短縮 3桁 `#RGB` は対象外（最初はシンプルさ優先）。必要になれば拡張可能。
