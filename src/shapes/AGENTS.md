この AGENTS.md は `src/shapes/` 配下に適用されます。

目的/役割
- プリミティブ形状（`BaseShape` 派生）とレジストリ（`shapes.registry`）。

外部とのつながり
- 依存可: `engine.core.Geometry`, `common/*`。
- 依存不可: `effects/*`, `engine/render/*`, `engine/pipeline/*`。

方針/Do
- `generate(**params) -> Geometry` は純粋関数的。変換は `BaseShape.__call__` で一括適用。
- 登録は `@shape`（BaseRegistry 準拠）を使用。

Don’t
- 描画・加工処理を混在させない。

テスト指針
- 代表パラメータの変化に対する頂点数・バウンディングの検証。

