この AGENTS.md は `src/shapes/` 配下に適用されます。

目的/役割
- プリミティブ形状（関数）とレジストリ（`shapes.registry`）。

外部とのつながり
- 依存可: `engine.core.Geometry`, `common/*`。
- 依存不可: `effects/*`, `engine/render/*`, `engine/runtime/*`。

方針/Do
- 形状関数は純粋関数的（副作用なし）で `Geometry` を返す（または `Geometry.from_lines()` 互換のポリライン列）。
- 変換（scale/rotate/translate）は `engine.core.geometry.Geometry` 側で適用。形状関数では行わない。
- 登録は `@shape`（BaseRegistry 準拠）を使用。

ドキュメンテーション（このパッケージ特例）
- 個々のシェイプ実装ファイルに「先頭ヘッダ（どこで・何を・なぜ）」は不要。
  - レジストリ/入口で役割を明記しているため、関数 docstring と `__param_meta__` を最小限に保つ。
  - ルート AGENTS.md の一般規約より、このディレクトリでは近接優先で本方針を優先する。

Don’t
- 描画・加工処理を混在させない。

テスト指針
- 代表パラメータの変化に対する頂点数・バウンディングの検証。
