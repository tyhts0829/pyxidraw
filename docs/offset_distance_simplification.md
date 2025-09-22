# offset: distance_mm の削除と distance への一本化 — 改善チェックリスト

目的
- API/UI の冗長性を排除し、単一パラメータ `distance` に統一する。
- 破壊的変更は可（AGENTS.md に基づく）。

変更範囲（最小）
- src/effects/offset.py: シグネチャから `distance_mm` を削除。実装・docstring・`__param_meta__` を同期。
- スタブ: `tools/gen_g_stubs` で `src/api/__init__.pyi` を再生成。
- ドキュメント: architecture.md（該当箇所があれば）を同期。
- テスト: `distance_mm` 参照があれば削除/置換（現時点で検索上は未使用想定）。

手順チェックリスト
- [x] offset シグネチャから `distance_mm` を削除し、実装の分岐（`distance_mm` 優先）を除去。
- [x] `offset.__param_meta__` から `distance_mm` を削除、説明文を `distance` のみに簡素化。
- [x] docstring から `distance_mm` 記述を削除。
- [x] 既定値の整合性を決定・反映（下記「確認事項 A」）。→ 15.0 に統一
- [x] スタブ再生成: `PYTHONPATH=src python -m tools.gen_g_stubs`。
- [x] 変更ファイル限定チェック: `ruff/black/isort/mypy`。
- [x] スモークテスト（該当エフェクト周辺のみ）。
- [x] architecture.md の同期（差分なし）。
- [x] 変更履歴/移行メモ（下記）。

確認事項（要承認）
- A. `distance` の既定値: 現コードは 15.0、doc には 5mm と記載。どちらに統一しますか？
  - 候補1: 15.0 のまま（挙動不変、ドキュメント更新）。
  - 候補2: 5.0 に変更（見た目マイルド、既存パイプラインの出力が変わる）。
- B. 完全削除で問題ないか（後方互換のための別名/警告は不要で良いか）。
  - 本リポは未配布のため、完全削除で簡素化が推奨。
- C. 実行時クリップ `MAX_DISTANCE=25.0` は現状維持で良いか（UI 側 RangeHint に合わせた runtime クランプ）。

影響とリスク
- `distance_mm` 指定を使う既存スクリプトがあれば実行時エラー（TypeError）。本リポ未配布のため影響は限定的。
- 既定値を 5.0 に変更する場合、既存サンプルの見た目が変わる可能性。

備考
- 実行時クリップと UI RangeHint は一致させる方針（AGENTS.md）。今回の変更では `distance` の RangeHint を維持し、`distance_mm` を撤廃。

移行メモ（2025-09-22）
- `effects.offset` から `distance_mm` を削除。距離指定は `distance` のみ。
- 既定値は 15.0 に統一（doc の 5mm 記述を修正）。
- 影響テストを更新済み（`tests/effects/test_offset_*`）。
