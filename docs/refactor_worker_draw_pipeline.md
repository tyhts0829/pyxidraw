# Worker 実装改善計画（draw 実行の共通化）

どこで: `src/engine/runtime/worker.py`

何を: ワーカの「draw 実行 → レイヤ正規化 → メトリクス差分 → RenderPacket 生成」の重複ロジック（子プロセス経路とインライン経路）を共通ヘルパへ抽出し、例外ハンドリング方針を一元化する。

なぜ: 可読性・保守性の向上、仕様差分の混入リスク低減、テスト容易性の向上。

関連箇所（重複している主な範囲）
- 子プロセス経路: `src/engine/runtime/worker.py:170` 付近（`_WorkerProcess.run()` 内）
- インライン経路: `src/engine/runtime/worker.py:240` 付近（`WorkerPool.tick()` 内）

## スコープ
- 共通化の対象:
  - CC/Parameter スナップショット適用（前処理/後処理）
  - メトリクス snapshot（before/after）取得と HIT/MISS 差分判定
  - `draw_callback(t)` の呼び出しと `_normalize_to_layers()` によるレイヤ正規化
  - `RenderPacket` 生成とキュー投入（呼び出し元が行う push は維持）
- 非対象（No-op/現状維持）:
  - Queue の生成・close、プロセス/スレッド起動・join 処理
  - `StreamReceiver` 側の処理
  - `_normalize_to_layers()` の中身（既存の共通ロジックを利用する）

## 設計方針（提案）
- `worker.py` 内に純粋関数スタイルの内部ヘルパを追加:
  - シグネチャ案:
    ```py
    def _execute_draw_to_packet(
        *,
        t: float,
        frame_id: int,
        draw_callback,
        apply_cc_snapshot,            # Callable[[Mapping[int,float]|None], None] | None
        apply_param_snapshot,         # Callable[[Mapping[str,object]|None, float], None] | None
        param_overrides,              # Mapping[str, object] | None
        metrics_snapshot,             # Callable[[], Mapping[str, Mapping[str,int]]] | None
    ) -> tuple[RenderPacket | None, WorkerTaskError | None]
    ```
  - 責務: 上記 1 フレーム分の一連の処理を実行し、`RenderPacket` か `WorkerTaskError`（どちらか一方）を返す。
  - 例外ハンドリング: ヘルパ内部で吸収して `WorkerTaskError` に変換。呼び出し側は戻り値を見てキューへ put するだけにする。

## 詳細ステップ
1) ヘルパ追加（実装）
   - CC/Parameter の適用（前処理）
   - メトリクス `before` 取得（失敗は None）
   - `draw_callback(t)` 実行
   - `_normalize_to_layers()` で `packet_geom`/`packet_layers` へ正規化
   - Parameter ランタイムのクリア（`apply_param_snapshot(None, 0.0)` を安全に呼ぶ）
   - メトリクス `after` 取得と差分フラグ生成（MISS 優先、HIT 二番手）
   - `RenderPacket` を生成
   - 例外時は `WorkerTaskError(frame_id, exc)` を返す

2) 呼び出し側の置換
   - 子プロセス経路（`_WorkerProcess.run`）: 既存ロジックをヘルパ呼び出しに差し替え
   - インライン経路（`WorkerPool.tick` の inline ブランチ）: 同上

3) ロギング方針
   - 例外はヘルパ内部で `logging.getLogger(__name__).exception("[worker] stage=...", ...)` を 1 箇所で出力
   - 呼び出し側では追加ログを行わない（多重出力防止）

4) 型と依存
   - 新規ヘルパは `worker.py` 内に留め、外部モジュール追加はしない（API 変更なし）
   - 既存の `RenderPacket`/`WorkerTaskError`/`_normalize_to_layers` を利用

5) テスト / 検証
   - 単体: `metrics_snapshot=None`/`apply_*_snapshot=None`/`param_overrides=None` の組合せで正常系/例外系を関数単位でテスト（関数抽出によりテスト容易になる）
   - 挙動互換: 既存デモ（`main.py`）実行で HUD の HIT/MISS 表示・描画の後方互換を人手確認
   - `pytest -q -k runtime`（対象テストがあれば）

6) ドキュメント/コメント
   - `worker.py` のモジュール先頭 docstring に「共通化ヘルパの役割」を 1 行追加
   - `architecture.md` への追記は不要（外部公開 API 変更なし）

## リスクと緩和
- リスク: ヘルパ抽出により subtle な順序依存が変化する可能性
  - 緩和: 抽出前後で実行順を厳密に踏襲（コードをほぼ移送）、差分はログのみ
- リスク: HUD の MISS/HIT 判定条件の退行
  - 緩和: 現行の MISS 優先規則をコメントとテストで固定

## 変更チェックリスト（承認後に着手）
- [ ] `_execute_draw_to_packet` を追加（仕様通り・型注釈）
- [ ] `_WorkerProcess.run` の重複ロジックを置換
- [ ] `WorkerPool.tick` の inline ブランチを置換
- [ ] 例外ログの重複を削減（呼び出し側の `logger.exception` を削除/最小化）
- [ ] 簡易ユニットテスト（可能なら）を追加/更新
- [ ] 目視リグレッション（デモ起動、HUD の cache status が更新されること）

## 事前確認事項（要回答）
1) 例外ログの出力レベルは現状通り `exception`（stacktrace 付き）で問題ないか。
2) ヘルパの戻り値を `RenderPacket | WorkerTaskError` にする方針で良いか（None は返さない）。
3) ヘルパは `worker.py` 内ローカルに留め、外部公開はしない方針で良いか。

承認いただければ、チェックリストに沿って実装を進め、進捗は本ファイルに反映（チェックを付与）します。

