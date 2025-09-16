この AGENTS.md は `src/engine/runtime/` 配下に適用されます。

目的/役割
- 非同期生成（`WorkerPool`）と受信（`StreamReceiver`）、二重バッファ（`SwapBuffer`）によるランタイム制御。

外部とのつながり
- 入出力は `Geometry` のみ。`engine/render/` へは `SwapBuffer` を介して連携。
- 依存不可: `effects/` の関数を直接呼ばない（ユーザ `draw()` の責務）。

方針/Do
- 例外は文脈付きで伝搬（`WorkerTaskError`）。最新フレーム優先で古い結果は破棄。

Don’t
- GPU 依存を持ち込まない。
