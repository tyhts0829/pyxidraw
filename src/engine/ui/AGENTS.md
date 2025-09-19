この AGENTS.md は `src/engine/ui/` 配下に適用されます。

目的/役割
- HUD/オーバーレイ等の軽量 UI とメトリクス表示。

外部とのつながり
- 依存可: `engine.render` からの描画フック、`engine.runtime` のバッファ参照（計測用）。
- 依存不可: 幾何処理や I/O への直接依存。

- 描画コールバックとメトリクス更新の順序を管理し、副作用を UI 層内に閉じ込める。
- `parameters` サブパッケージでは `FunctionIntrospector`/`ParameterValueResolver` を介してメタ情報抽出と値適用を分離し、`ParameterRuntime` はオーケストレーションのみに専念する。
