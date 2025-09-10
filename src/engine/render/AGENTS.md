この AGENTS.md は `src/engine/render/` 配下に適用されます。

目的/役割
- `Geometry` を VBO/IBO に変換し、ModernGL で線描画する。

外部とのつながり
- 入力: `engine.pipeline.SwapBuffer`（front バッファ）。
- 依存可: `util.constants.PRIMITIVE_RESTART_INDEX`。
- 依存不可: `effects/*`, `shapes/*`, `api/*`。

方針/Do
- 1 ドローに集約（primitive restart）。GPU リソースのライフサイクルを明示管理。

Don’t
- 形状生成・加工のロジックを持ち込まない。

