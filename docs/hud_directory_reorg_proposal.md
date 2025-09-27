# HUD 関連モジュールのディレクトリ構成（再検討・提案）

目的
- HUD 関連の「収集（metrics/sampling）」と「表示（overlay）」を整理し、責務と依存方向を明確化する。
- 将来の HUD 項目追加・計測追加を容易にし、パフォーマンスオプションの制御点を一貫化する。

旧構成（参考、再編前）
- 表示: `src/engine/ui/overlay.py` — `MetricSampler.data` を pyglet で描画。
- 収集(ランタイム/HUD): `src/engine/ui/monitor.py` — FPS/VERTEX/CPU/MEM を一定間隔でサンプリング。
- 収集(キャッシュ診断): `src/diagnostics/metrics.py` — shape/effect のキャッシュ累計スナップショット。
- 設定: `src/engine/ui/hud/config.py`（HUDConfig）、`src/engine/ui/hud/fields.py`（キー名定義）。

課題
- 「metrics」という語が `diagnostics/metrics.py` と `ui/monitor.py` の両方に登場し紛らわしい。
- HUD 向けの収集（CPU/MEM/FPS/頂点）は UI/hud に属する一方、キャッシュ診断は HUD 専用ではない（将来 CLI/ログにも流用可）。
- 依存方向の明示: `engine` は `api` を参照しない。`diagnostics` は `api` を直接参照してもよいが、将来的には下層（effects/shapes）に寄せたい。

提案（ディレクトリ構成）
選定更新: engine → api 依存禁止ルールにより、`engine/runtime/cache_snapshot.py` 方式は不採用。
現行は `api.sketch` 内のトップレベル関数としてスナップショットを実装し、Worker へ関数注入する。
- HUD（UI 側）: `src/engine/ui/hud/`
  - `config.py`（現行）: HUDConfig 定義
  - `fields.py`（現行）: 表示キー名
  - `sampler.py`（新）: 既存 `ui/monitor.py` の `MetricSampler` を移動
  - `overlay.py`（新）: 既存 `ui/overlay.py` を移動（将来 `render.py` へ分割も可）
  - 役割: HUD 表示に必要なランタイム指標のサンプリングとレンダリング。`diagnostics` の関数は“任意注入”で使用。
- 診断（横断ユーティリティ）: `src/diagnostics/`
  - `cache_snapshot.py`（改名）: 既存 `metrics.py` を HUD 依存の無い「キャッシュスナップショット API」に改名
  - `process.py`（任意・将来）: CPU/MEM 等の psutil ベースユーティリティ（UI 抜きでも利用可能な形）
  - 役割: UI に依存しない計測関数群。spawn 安全なトップレベル関数として提供。

代替案（diagnostics を廃止して単一モジュールへ集約）
- 目的: キャッシュヒット計測しか無い現状で専用ディレクトリを増やさない。最小で分かりやすい配置にする。
- 案A（推奨）: `src/engine/runtime/cache_snapshot.py`
  - `snapshot_counters()` をここに置く。Worker で使う差分計測の責務に近い（ランタイム寄り）。
  - `api.sketch` から関数を注入する流れは維持（engine は api を参照しない）。
  - 依存: `effects`/`shapes` に直接依存（API 経由なし）。
  - 必要な前提変更（別タスク）:
    - `api.effects.global_cache_counters` を `effects.metrics:global_cache_counters` に移設（または再エクスポート）。
    - `ShapesAPI.cache_info()` 相当の形状キャッシュ統計を `shapes.metrics:shape_cache_counters` として提供（API 依存を排除）。
- 案B: `src/effects/metrics.py` に統合
  - effect 側の累計取得は自然だが、shape 側の統計も参照するため `effects`→`shapes` の横断依存が生じる。
  - 境界が曖昧になりやすいので案Aより劣後。
- 案C: `src/engine/ui/hud/sampler.py` に同居
  - HUD 近接で分かりやすいが、UI から domain（effects/shapes）を直接参照することになる。
  - 将来 HUD OFF の CLI/ログ出力で再利用するときに場所が不自然。案Aより劣後。

依存と責務の境界（本提案の前提）
- `engine.ui.hud` → `engine.core`, `engine.runtime`, `diagnostics`（注入された関数を利用）
- `diagnostics` → `effects`, `shapes` に直接依存（API 経由にしない）。
  - 目的: engine→api の逆参照が生じる経路を完全排除し、依存方向を一貫化。
  - 実装注記: `snapshot_counters` は `effects.registry`/`shapes.registry` または各モジュールの公開ユーティリティを直接参照して集計する。
- `engine.runtime.worker` は `metrics_snapshot: Callable | None` を受け取り、診断の存在を知らない（既存踏襲）。

命名ポリシー
- HUD 用の収集は「sampler」、UI 非依存の累計/スナップショットは「snapshot」を用語として使い分け。
- 表示キーは `hud/fields.py` に集約。Overlay はキーの有無だけを見て描画。

移行プラン（コード変更が必要な場合の手順）
1) `src/engine/ui/monitor.py` → `src/engine/ui/hud/sampler.py`（移動・import 更新）
2) `src/engine/ui/overlay.py` → `src/engine/ui/hud/overlay.py`（または `render.py`）
3) キャッシュ計測の配置を以下のいずれかで決定:
   - 案A（推奨）: `src/engine/runtime/cache_snapshot.py` を新設し、`snapshot_counters()` を移す。
   - 案B: `src/effects/metrics.py` に統合。
   - 案C: `src/engine/ui/hud/sampler.py` に併置。
4) 参照先の整理（API 依存の解消）:
   - `api.effects.global_cache_counters` → `effects.metrics.global_cache_counters` に移設または再エクスポート。
   - `ShapesAPI.cache_info()` 依存を、`shapes.metrics.shape_cache_counters()` に置換。
5) `src/api/sketch.py` の import を新配置に合わせて更新（注入は既存通り）。
6) ドキュメント: `architecture.md` と関連 docs の参照を更新（依存方向図を `engine.runtime ↔ effects/shapes`に整合）。

後方互換の扱い（選択肢）
- クリーン移行: 互換リダイレクトを置かず、参照をすべて更新（推奨・現リポは破壊的変更可）
- 段階移行: 旧パスに薄い再エクスポートを短期間だけ配置（要削除期限）

将来拡張のための指針
- 新たな HUD 項目は `hud/fields.py` にキーを追加し、`MetricSampler` で値を埋める。Overlay は変更不要。
- UI 非依存の計測（GPU 統計、I/O レイテンシ等）は `diagnostics/` に追加し、`api.sketch` から Worker や HUD へ関数注入する。
- `diagnostics` は `effects`/`shapes` を直接参照する（API を経由しない）。Breaking 変更が必要なら別計画で段階的に導入する。

DoD（完了条件）
- HUD 関連ファイルが `engine/ui/hud/` に集約され、`diagnostics/` は UI 非依存の計測関数のみを持つ。
- 依存方向が明確（engine → diagnostics は関数注入、engine は api を参照しない）。
- Docs が新構成を反映し、`ruff/black/isort/mypy` が緑。

メモ
- 本提案はドキュメントのみ。コード変更は含めない（別 PR/タスクで適用）。
