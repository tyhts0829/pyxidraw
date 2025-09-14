この AGENTS.md は `src/api/` 配下に適用されます（近接優先）。

目的/役割
- 公開 API のエントリポイント。
- `G`（形状ファクトリ）、`E.pipeline`（エフェクトパイプライン）、`run`/`run_sketch` の薄いファサードを提供。

外部とのつながり（依存関係）
- 依存可: `engine.core.Geometry`、`effects.registry`、`shapes.registry`、`api.runner`。
- 依存原則: `api/*` から `engine/*` への依存は最小限（公開 API の薄いファサードに限定）。
- 例外（runner.py のみ許可）:
  - `api/runner.py` は「実行オーケストレータ」として、以下のサブシステムへ遅延 import で依存してよい。
    - `engine.render/*`、`engine.pipeline/*`、`engine.ui/*`、`engine.io/*`、`engine.monitor/*`
  - 条件:
    - 依存は関数内の遅延 import とし、モジュールトップで import しない（ヘッドレス/未導入環境の互換確保）。
    - `engine/*` 側から `api/*` を参照しない（依存の一方向性を維持）。
    - `api/runner.py` は `engine/*` の詳細型を再エクスポートしない（公開 API の表面を汚さない）。
  - 上記以外の `api/*` モジュールは従来どおり `engine.render/*` および `engine.pipeline/*` への直接依存を禁止。

方針/Do
- 変換・加工ロジックを持ち込まず、委譲に徹する。
- シリアライズ/検証（`to_spec/from_spec/validate_spec`）はここでまとめる。

Don’t
- ハードウェア I/O を直接扱わない（MIDI/GL へは `engine` 経由）。
- 重い依存のモジュールトップ import。

テスト指針
- 入口 API のみの単体テスト（パラメータ透過、例外、LRU 動作）。
