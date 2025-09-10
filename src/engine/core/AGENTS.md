この AGENTS.md は `src/engine/core/` 配下に適用されます。

目的/役割
- 統一幾何 `Geometry` とフレーム制御（`Tickable`/`FrameClock`/`RenderWindow`）。

外部とのつながり
- 依存可: `common/*`, `util/constants.py`。
- 依存不可: `effects/*`, `shapes/*`, `engine/render/*` の実装詳細（`RenderWindow` は pyglet 依存の薄いラッパ）。

方針/Do
- `Geometry` は唯一の幾何表現。変換は純関数で提供。

Don’t
- I/O や GPU への直接依存を増やさない。

