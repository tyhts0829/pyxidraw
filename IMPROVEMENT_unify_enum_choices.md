# 改善計画: enum 判定を `choices` に統一

目的
- enum（カテゴリ変数）パラメータは `choices` の有無だけで判定し、`type: "string"` には依存しない形に統一する。
- ドキュメント/実装の一貫性を高め、UI 実装側の分岐を単純化する。

方針
- enum = `__param_meta__` に `choices` がある場合のみ。
- 自由文字列（非 enum）= `type: "string"` かつ `choices` 無し。
- 既存の enum メタからは `type: "string"` を削除（`choices` のみで表現）。

対象（修正予定ファイル）
- src/effects/extrude.py（`center_mode`）
- src/effects/offset.py（`join`）
- src/effects/twist.py（`axis`）
- src/effects/fill.py（`mode`）
- src/shapes/text.py（`align`）
- src/shapes/attractor.py（`attractor_type`）
- src/shapes/polyhedron.py（`polygon_type`）

作業チェックリスト
- [x] 上記各ファイルの `__param_meta__` から enum 項目の `type: "string"` を削除
- [x] grep（`rg '"type"\s*:\s*"string"' src/{effects,shapes}`）で取りこぼしが無いことを確認（自由文字列 `text.font` のみ残存）
- [x] 影響ドキュメントを同期（AGENTS.md の短記: 「enum は choices の有無で判定」）
- [x] 変更ファイルに限定して ruff/black/isort を実行（mypy は既存の別件エラーで失敗）
- [ ] 最小テスト: `pytest -q -m smoke` と `tests/ui/parameters`（enum が落ちないことの確認）
- [ ] スタブ再生成ドライランで差分が実質なし（コメントの `type=string` 減少のみ）を確認

受け入れ条件（DoD）
- 対象すべてで `choices` のみの定義に統一（`type: "string"` 併記なし）
- 既存の機能/テストが維持される（挙動変更なし）
- リポ規約（enum 判定は `choices` ベース）が docs に反映

補足
- `tools/gen_g_stubs.py` は `choices` のみでもコメント生成が可能（`type` は任意）
- `ParameterValueResolver` は `choices` 有無で `enum` を判定可能（現状ロジックと整合）

質問（最終確認）
1) enum 項目からの `type: "string"` 削除で問題ないか（全削除）
2) 自由文字列の箇所（例: `text.font`）はそのまま `type: "string"` 維持でよいか

準備が整い次第、このチェックリストに沿って実装します。
