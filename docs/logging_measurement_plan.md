# ログ計測 戦略と実装チェックリスト（ドラフト）

本書は「src 配下のモジュール群に logging を用いた計測を適用する」ための方針と、段階的な実装チェックリストを示すドラフト。実装前の合意形成を目的とする。

- 対象: `src/` 全体（優先: `engine/runtime`, `engine/render`, `engine/core`, `effects`, `shapes`, `api`）
- 依存: 標準ライブラリ `logging` のみ（追加依存なし）
- 出力: 既定は stderr（コンソール）。必要に応じてファイルハンドラ追加に拡張可能
- 形式: 既定は logfmt（`key=value`）を推奨。将来的に JSON を任意選択可能
- 命名: `logging.getLogger(__name__)`（パッケージ名は `pyxidraw`）
- 方針: シンプル・必要十分・安全な実装を優先（AGENTS.md に準拠）

---

## 目的（What）
- 観測可能性の向上: 実行フローと異常の可視化
- パフォーマンス計測: フレーム時間、重い関数の所要時間、スループット
- トラブルシュート簡素化: 失敗時の最小限の状況情報（コンテキスト）を付与
- 負荷制御: ログ量・オーバーヘッドの制御（しきい値・サンプリング）

## 原則（Why）
- 複雑化を避け、標準 `logging` の素朴な構成で始める
- 「概要は INFO」「詳細は DEBUG」へ明確に分離
- 高頻度領域はサンプリングとしきい値で抑制
- 追加依存なし（Ask-first 違反を回避）

## 出力レベルと使い分け
- INFO: ライフサイクル（開始/停止/設定解決）、1 フレーム単位の集約メトリクス
- DEBUG: 重い処理の詳細タイミング、キャッシュヒット率、内部状態の軽量要約
- WARNING: リトライやフォールバック、想定内の例外経路
- ERROR/CRITICAL: 失敗（`exc_info=True` でトレースバック）

## 形式とフィールド（logfmt 推奨）
- 例（フレーム集約）: `event=frame_summary frame=123 dt_ms=16.7 shapes=42 effects=5 cache_hit=0.83`
- 例（関数計測）: `event=duration name=geometry.flatten_ms dur_ms=4.12 frame=123 shape=grid`
- 共通フィールド（推奨）:
  - `event`: 短いイベント名（必須）
  - `frame`: フレーム番号 or `t` の近似（整数）
  - `dur_ms`: 経過時間 [ms]（`time.perf_counter_ns` 由来）
  - `name`: 関数や処理名
  - `shape` / `effect`: 処理対象の種別識別子（あれば）
  - `worker`/`thread`: 実行スレッド/ワーカー ID（任意）
  - `err`: 例外要約（ERROR 時）

注: 配列や巨大オブジェクトは出さない（サイズや件数など要約のみ）。

## 実装戦略（How）
1) 入口構成（最小）
   - `main.py` で一度だけ `setup_logging()` を呼び出し、環境変数で挙動を切替
     - `PXD_LOG_LEVEL=INFO|DEBUG`（既定: `INFO`）
     - `PXD_LOG_FORMAT=logfmt|json`（既定: `logfmt`）
     - `PXD_LOG_SLOW_MS=<float>`（既定: `8.0`。超過時に duration を INFO で出す）
     - `PXD_LOG_SAMPLING=1|N`（既定: `1`。`N>1` なら `1/N` で DEBUG をサンプリング）

2) ユーティリティの用意（`util/logging_utils.py` 新設）
   - `setup_logging()`：Formatter/Handler/Filter を構成
   - `LogContext`：`contextvars.ContextVar` で `frame`/`t` 等を伝搬
   - `with log_duration(name, level=DEBUG, slow_ms=None, sample=None, extra=None)`：計測 CM
   - `@log_calls(name=None, ...)`：軽量デコレータ（引数/戻り値は出さない）
   - `LoggingAdapter`：`extra` の既定（`frame`, `worker` 等）を注入
   - 実装は「軽量・副作用最小・計算コスト抑制」を徹底（`isEnabledFor` で枝刈り）

3) 計測ポイント（優先度順）
   - ランタイム: `engine/runtime/*` のフレーム駆動（1 フレーム 1 行の集約 INFO）
   - レンダリング: `engine/render/*` のアップロード/描画パス（各パスの duration）
   - コア幾何: `engine/core/geometry.py` の重い変換（flatten/merge/offset 等）
   - エフェクト/シェイプ: 生成・変形の主要ステップ（しきい値/サンプリング前提）
   - I/O: G-code エクスポートのジョブ開始/完了/失敗

4) ログ量制御
   - 既定は「フレーム集約 INFO」+「しきい値超過のみ INFO」+「詳細 DEBUG（必要時のみ）」
   - DEBUG は `PXD_LOG_SAMPLING` と `slow_ms` で抑制

5) エラー方針
   - 例外は `logger.exception("event=... ...")`（logfmt メッセージ先頭に `event=`）
   - 大きいデータはダンプしない。識別子と件数のみ

6) 型/スタイル
   - 型注釈必須（Python 3.10）。ruff/mypy の対象に含める
   - 既存の `logging.getLogger(__name__)` パターンと整合

## イベント・タクソノミ（案）
- `app_start`, `app_stop`
- `frame_start`, `frame_summary`（`dt_ms`, `shapes`, `effects`, `cache_hit` など）
- `render_upload`, `render_draw`, `render_present`
- `geom_op`（`name`, `dur_ms`, `lines_in/out`）
- `effect_apply`, `shape_generate`
- `export_job_start`, `export_job_done`, `export_job_fail`
- `fallback`（MIDI/UI 等のフォールバック通知）

## 最小セット実装の受け入れ基準（DoD）
- `main.py` に最小初期化（`setup_logging()`）を追加
- `util/logging_utils.py` を追加（上記ユーティリティ一式）
- ランタイムのフレーム集約 INFO を実装
- 代表的な 2 箇所（レンダ/幾何）で `log_duration` を適用
- `ruff/mypy/pytest -q -m smoke` が変更ファイルでグリーン

## 段階導入ロードマップ
- Phase 1（最小）：入口 + フレーム集約 + 2 箇所の `log_duration`
- Phase 2（拡張）：レンダ/幾何の主要ホットスポットへ適用拡大、エラー統一
- Phase 3（任意）：JSON 出力/ファイルハンドラ/簡易トレース ID（`trace_id`）

## 要確認事項（ご相談）
1) 既定フォーマット: `logfmt`（key=value）でよいか。JSON はオプションとするか
2) しきい値の既定: `PXD_LOG_SLOW_MS=8.0` ms で許容か
3) サンプリング既定: `PXD_LOG_SAMPLING=1`（無効）で開始してよいか
4) フレーム集約の項目: `dt_ms/shapes/effects/cache_hit` で十分か
5) ファイル出力: 初期はコンソールのみでよいか（ファイルは後日）
6) `architecture.md` への反映は Phase 1 完了時に追記でよいか

## 実装チェックリスト（着手前）
- [ ] 入口: `main.py` に `setup_logging()` 呼び出し
- [ ] ユーティリティ: `util/logging_utils.py` 追加（型付き）
- [ ] ランタイム: 1 フレーム集約 INFO（`engine/runtime`）
- [ ] レンダ: 主要パスの `log_duration`（`engine/render`）
- [ ] 幾何: ホットスポットの `log_duration`（`engine/core`）
- [ ] I/O: エクスポート開始/完了/失敗を INFO/ERROR
- [ ] ログ量制御: `slow_ms` と `sampling` を適用
- [ ] ドキュメント: `architecture.md` に最小追記（Phase 1 完了時）
- [ ] 検証: 変更ファイルに対する ruff/mypy/pytest（smoke）

---

補足（安全性）
- 個人情報/生データを出さない。サイズや件数などのメタ情報のみ
- 重いフォーマット/計算は `isEnabledFor` ガードの内側でのみ実行
- 乱数・時間起因の非決定性はログ上で最小化（必要に応じて丸め）

このドラフトで進めてよければ、Phase 1 の実装に着手します。修正・希望があればご指示ください。

