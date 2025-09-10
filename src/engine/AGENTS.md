この AGENTS.md は `src/engine/` 配下に適用されます（下位の AGENTS.md が優先）。

目的/役割
- 実行エンジン層の総称。Core/Pipeline/Render/UI/IO を包含するが、それぞれ責務を分離。

外部とのつながり
- 上位: `api/` からのみ直接呼び出される想定。
- 下位: `common/*`, `util/*` にのみ依存。

方針/Do
- サブシステム間は `Geometry` と小さなインターフェース（`Tickable` 等）で連携。
- ハードウェア依存（GL/MIDI）はサブディレクトリに局在化。

Don’t
- `effects/`・`shapes/` を import しない。

