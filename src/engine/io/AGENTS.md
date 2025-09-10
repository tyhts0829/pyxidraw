この AGENTS.md は `src/engine/io/` 配下に適用されます。

目的/役割
- MIDI デバイス検出・CC 値供給（`MidiService`）。

外部とのつながり
- 依存可: `mido`（遅延 import）、`util.utils.load_config`。
- 出力: `MidiService.snapshot()` による CC マップ（`Mapping[int,float]`）。

方針/Do
- モジュールトップで重依存を import しない（遅延 import）。
- 失敗時は厳格/緩和モードで分岐（呼び出し側で制御）。

Don’t
- 幾何・GPU・UI へ依存しない。

