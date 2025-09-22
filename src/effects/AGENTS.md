この AGENTS.md は `src/effects/` 配下に適用されます。

目的/役割
- `Geometry -> Geometry` の純関数エフェクト群とレジストリ（`effects.registry`）。

外部とのつながり
- 依存可: `engine.core.Geometry`, `util.geom3d_ops`, `common/*`。
- 依存不可: `engine.render/*`, `engine.runtime/*`, `api/*`（処理は委譲/入口側）。

方針/Do
- 入出力は `Geometry` に統一。副作用を避け決定的に。
- 登録は `@effect`（BaseRegistry 準拠）を使用。

ドキュメンテーション（docstring 規約）
- 関数 docstring（ユーザー向け）
  - 目的: IDE/補完/ヘルプで参照される最小限の利用情報。
  - 構成: 先頭1行の要約 + Parameters のみ。
  - Parameters に含める: 型、既定値、単位、範囲/クランプ、no-op 条件など“使い方の事実”。
  - 含めない: Returns/Notes/Examples/実装詳細/性能メモ/内部スイッチ。
  - 形式: NumPy スタイル、日本語の事実記述（主語省略・終止形）。
- モジュール docstring（必要な場合のみ、開発者向け最小ヘッダ）
  - 内容: どこで・何を・なぜ（3〜5行）。実装メモの要点や環境スイッチ（例: `PXD_USE_NUMBA_*`）は短く記す。
  - 詳細: 設計/最適化/検証/トレードオフは docs 側（例: `docs/effects/<effect>.md`）や ADR/architecture.md に記載。
- 単一情報源/同期
  - `__param_meta__` と関数 docstring の範囲・既定値・単位を一致させる。
  - パラメータ変更時は docstring と `__param_meta__` を同時更新。公開 API 影響時はスタブ再生成とテスト確認。
- テンプレ（例）
  
  先頭1行: 「連続線を破線に変換。」
  
  Parameters
  ----------
  g : Geometry
      入力ジオメトリ。各行が 1 本のポリラインを表す（`offsets` で区切る）。
  foo : float, default 1.0
      単位 mm。許容 [0, 100]。0 で no-op。
  bar : str, default 'round'
      `'mitre'|'round'|'bevel'` を指定。

Don’t
- I/O, GPU, ウィンドウや MIDI へ依存しない。

テスト指針
- 小さな入力での安定性・境界値・ゼロ入力時のコピー返却など。
