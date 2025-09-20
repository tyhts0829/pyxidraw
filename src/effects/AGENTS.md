この AGENTS.md は `src/effects/` 配下に適用されます。

目的/役割
- `Geometry -> Geometry` の純関数エフェクト群とレジストリ（`effects.registry`）。

外部とのつながり
- 依存可: `engine.core.Geometry`, `util.geom3d_ops`, `common/*`。
- 依存不可: `engine.render/*`, `engine.runtime/*`, `api/*`（処理は委譲/入口側）。

方針/Do
- 入出力は `Geometry` に統一。副作用を避け決定的に。
- 登録は `@effect`（BaseRegistry 準拠）を使用。

ドキュメンテーション（このパッケージ特例）
- 個々のエフェクト実装ファイルに「先頭ヘッダ（どこで・何を・なぜ）」は不要。
  - 入口やレジストリ側で意図を説明済みのため、本体は関数 docstring と `__param_meta__` を最小に保つ。
  - ルート AGENTS.md の一般規約に対するローカル上書き（近接優先）。

Don’t
- I/O, GPU, ウィンドウや MIDI へ依存しない。

テスト指針
- 小さな入力での安定性・境界値・ゼロ入力時のコピー返却など。
