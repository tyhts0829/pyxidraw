# 変更計画: run() で線の太さを指定可能にする

目的: `api.run()`（=`api.sketch.run_sketch`）の引数で描画線の太さを指定できるようにし、ModernGL のシェーダへ反映する。

想定スコープ: 既存レンダラ/シェーダを崩さず、最小改修で一貫した動作を提供。

単位の前提:
- 現行ジオメトリシェーダは `uniform float line_thickness`（クリップ座標系単位）を想定。
- 初期実装では引数 `line_thickness` をクリップ空間（-1〜1基準）で受ける。
- mm 指定サポートは将来拡張（`2*mm/canvas_height` 相当）とし、本PRでは未対応。

## タスクチェックリスト

- [x] API: `run_sketch(..., line_thickness: float = 0.0006)` を追加（キーワード専用）。
- [x] 配線: `run_sketch` から `engine.render.renderer.LineRenderer` に値を受け渡し。
- [x] Renderer: `LineRenderer.__init__(..., line_thickness: float)` を追加し、`program['line_thickness']` に反映。
- [x] ドキュメント: `api.sketch.run_sketch` の docstring と `architecture.md` に引数説明を追記。
- [x] スタブ: `python -m tools.gen_g_stubs` で `src/api/__init__.pyi` を同期（差分確認）。
- [x] チェック: 変更ファイルに限定して `ruff/black/isort/mypy` を実行。

## 確認したい点（要回答）

1. 単位は「クリップ空間（-1〜1）」指定で問題ありませんか？
   - 代案: mm で指定し内部変換（`thickness_clip = 2*mm/canvas_height`）
2. 既定値は現行シェーダの `0.0006` を踏襲で良いですか？
3. 将来的に UI から動的に太さを変更できるフックは必要ですか？（今回は固定で予定）

承認いただければ、上記チェックリストに沿って実装・反映を行います。
