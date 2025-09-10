この AGENTS.md は `src/engine/monitor/` 配下に適用されます。

目的/役割
- 実行時メトリクス（FPS/頂点数/CPU/MEM）のサンプリング。

外部とのつながり
- 入力: `engine.pipeline.SwapBuffer`。
- 出力: HUD 等の表示層へ辞書提供のみ（副作用小）。

方針/Do
- 一定間隔での軽量計測。表示順・単位をここで正規化。

