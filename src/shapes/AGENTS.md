この AGENTS.md は `src/shapes/` 配下に適用されます。

目的/役割
- プリミティブ形状（関数）とレジストリ（`shapes.registry`）。

外部とのつながり
- 依存可: `engine.core.Geometry`, `common/*`。
- 依存不可: `effects/*`, `engine/render/*`, `engine/runtime/*`。

方針/Do
- 形状関数は純粋関数的（副作用なし）で `Geometry` を返す（または `Geometry.from_lines()` 互換のポリライン列）。
- 変換（scale/rotate/translate）は `engine.core.geometry.Geometry` 側で適用。形状関数では行わない。
- 登録は `@shape`（BaseRegistry 準拠）を使用。

ドキュメンテーション（docstring 規約）
- 関数 docstring（ユーザー向け）
  - 目的: IDE/補完/ヘルプで参照される最小限の利用情報。
  - 構成: 先頭1行の要約 + Parameters のみ。
  - Parameters に含める: 型、既定値、単位、範囲/クランプ、no-op 条件など“使い方の事実”。
  - 含めない: Returns/Notes/Examples/実装詳細/性能メモ/内部スイッチ。
  - 形式: NumPy スタイル、日本語の事実記述（主語省略・終止形）。
- モジュール docstring（必要な場合のみ、開発者向け最小ヘッダ）
  - 内容: どこで・何を・なぜ（3〜5行）。実装メモの要点は短く記す。
  - 詳細: 設計/最適化/検証/トレードオフは docs 側（例: `docs/shapes/<shape>.md`）や ADR/architecture.md に記載。
- 単一情報源/同期
  - `__param_meta__` と関数 docstring の範囲・既定値・単位を一致させる。
  - パラメータ変更時は docstring と `__param_meta__` を同時更新。公開 API 影響時はスタブ再生成とテスト確認。
- テンプレ（例）
  
  先頭1行: 「半径 r の円を生成。」
  
  Parameters
  ----------
  r : float, default 50.0
      単位 mm。許容 [0, 500]。0 で空形状。
  segments : int, default 64
      円弧近似の分割数。

Don’t
- 描画・加工処理を混在させない。

テスト指針
- 代表パラメータの変化に対する頂点数・バウンディングの検証。
