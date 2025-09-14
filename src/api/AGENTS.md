この AGENTS.md は `src/api/` 配下に適用されます（近接優先）。

目的/役割
- 公開 API のエントリポイント。
- `G`（形状ファクトリ）、`E.pipeline`（エフェクトパイプライン）、`run`/`run_sketch` の薄いファサードを提供。

外部とのつながり（依存関係）
- 依存可: `engine.core.Geometry`、`effects.registry`、`shapes.registry`、`api.runner`。
- 依存不可: `engine.render/*` の具体 GPU 実装や `engine.pipeline/*` の詳細へ直接依存しない。

方針/Do
- 変換・加工ロジックを持ち込まず、委譲に徹する。
- シリアライズ/検証（`to_spec/from_spec/validate_spec`）はここでまとめる。

Don’t
- ハードウェア I/O を直接扱わない（MIDI/GL へは `engine` 経由）。
- 重い依存のモジュールトップ import。

テスト指針
- 入口 API のみの単体テスト（パラメータ透過、例外、LRU 動作）。
