# 安全終了（ESC/ウィンドウクローズ）実装改善計画

目的: ESC で描画ウィンドウを閉じた際に、ターミナルのエラー吐き出しや子プロセス残存を無くし、常に静かで確実に終了できるようにする。

## 完了定義（DoD）
- ESC またはウィンドウ右上のクローズで終了したとき、標準出力/標準エラーに例外・トレースバックが出ない。
- `multiprocessing` の子プロセス/スレッドが残らない（`mp.active_children()==[]` 相当、`ps` で残存プロセスなし）。
- 録画/画像保存/エクスポートなどのバックグラウンド処理が安全に停止（.part ごみが残らない）。
- GUI（Dear PyGui）ドライバが停止し、終了直前に追加の `tick()` が走らない。
- 変更ファイルに対する `ruff/mypy/pytest -q -m smoke` が緑。

## 現状の終了フロー（抜粋・参照）
- ウィンドウクローズ: `src/api/sketch.py:512` の `on_close()` でワーカ停止、録画停止、MIDI保存、GPU解放、Parameter GUI の `shutdown()`、最後に `pyglet.app.exit()`。
- ワーカ停止: `src/engine/runtime/worker.py:319` の `WorkerPool.close()` で `None` センチネル投入 → `join(timeout=1.0)` → 生存時 `terminate()`、最後にキュー `close()`。
- DPG 駆動: `src/engine/ui/parameters/dpg_window.py:1315` 以降で `pyglet.clock.schedule_interval(self._tick, ...)` による駆動。停止は `:1335` の `_stop_driver()` で `unschedule()`。
- 録画品質モード: `src/api/sketch_runner/recording.py` でフレーム駆動の切替と `unschedule()` を実施。

観測される問題の可能性
- `on_close()` 実行中にフレーム駆動（`pyglet.clock` の `frame_clock.tick`）がまだ生きており、リソース解放後に `tick()` が走ってエラーを出す。
- `multiprocessing.Process(daemon=True)` のため、親終了時の強制終了でログや標準ストリームに絡むシャットダウン時例外が見えるケースがある。
- Export/G-code のバックグラウンドスレッドが `daemon=True` で強制終了し、.part が残る。
- シグナル（`SIGINT`/`SIGTERM`）受信時の経路がウィンドウクローズと統一されていない。

## 設計方針（シンプル・冪等）
- 終了処理は「駆動停止 → 生産停止 → 資源解放 → 退場」の順序で一箇所に集約し、冪等（多重呼び出し安全）にする。
- スケジューラ（`pyglet.clock`）の `unschedule()` を最初に行い、以降の `tick()` 実行を止めてからリソース解放する。
- ワーカープロセスは可能な限り `join()` によるグレースフル停止、間に合わない場合のみ `terminate()`。
- OS シグナル/`atexit` で同じシャットダウン関数を呼び、経路を統一。

## 実装タスク（チェックリスト）

1) シャットダウン調停（ShutdownCoordinator）の導入 [runtime]
- [ ] 追加: `src/engine/runtime/shutdown.py` を新設し、以下を提供
  - [ ] `register(name: str, func: Callable[[], None], order: int = 0)` 登録（順序つき）。
  - [ ] `shutdown(reason: str = "")` 一度だけ実行（冪等・例外握りつぶし）。登録順を `order` 降順で実行。
  - [ ] `is_shutting_down()` フラグ API。
- [ ] atexit/シグナル連携（後述）からも同一 `shutdown()` を呼ぶ。

2) ランナー（api.sketch）の順序見直し・統合
- [ ] 変更: `src/api/sketch.py:512` の `on_close()` の先頭で `pyglet.clock.unschedule(frame_clock.tick)` を呼ぶ。
- [ ] 変更: `on_close()` の個別 `try:` 群を `ShutdownCoordinator` 登録へ移し、`on_close()` は調停器を呼ぶだけに簡素化。
- [ ] 追加: `signal.signal(SIGINT/SIGTERM/SIGHUP, ...)` でウィンドウが在る場合は `rendering_window.close()`、無い場合は `ShutdownCoordinator.shutdown("signal")` を呼ぶ。
- [ ] 追加: `atexit.register(ShutdownCoordinator.shutdown)` を登録。
- [ ] 変更: ESC 押下（`src/api/sketch.py:452`）で `rendering_window.close()` に加えて、必要ならば「二度押しで強制終了」のガード（任意・デフォルトOFF）。

3) ワーカープロセスのクリーン終了強化
- [ ] 変更: `src/engine/runtime/worker.py:61` の `_WorkerProcess` 生成で `daemon=True` を解除（`super().__init__(daemon=False)`）。
- [ ] 変更: `WorkerPool.close()` の `join(timeout=1.0)` を可変（環境変数 `PXD_SHUTDOWN_TIMEOUT`、既定 1.0s）。
- [ ] 追加: `result_q` が `mp.Queue` の場合に `cancel_join_thread()` を呼ぶ（送出側終了時のハング防止）。
- [ ] 追加: `close()` が呼ばれた後の `tick()` は no-op（早期 return、現状維持を明確化）。

4) フレーム駆動の停止順序の明示
- [ ] 変更: 通常モード終了時に `pyglet.clock.unschedule(frame_clock.tick)` を必ず最初に行う。
- [ ] 変更: 品質モード終了経路（`src/api/sketch_runner/recording.py`）は現状で `unschedule()` 済みだが、`on_close()` からの二重呼び出しでも安全であることをコメントで保証。

5) Dear PyGui ドライバ停止の確実化
- [ ] 確認: `ParameterWindow._stop_driver()` は `unschedule()` を呼んでいる（`src/engine/ui/parameters/dpg_window.py:1335`）。
- [ ] 変更: `ParameterWindow.close()` 経由で必ず `_stop_driver()` → `destroy_context()` の順に実行されることを、`ParameterManager.shutdown()` 呼び出し順（`src/api/sketch.py:542-545`）の直前に明示コメントとして残す。

6) Export/G-code スレッドの停止（後方互換のまま改善）
- [ ] 追加: `ExportService` に `shutdown(timeout: float = 0.5)` を追加し、実行中ジョブに `cancel_event` を投げ、キューの `join()` を短時間待機。
- [ ] 変更: ランナー終了時（`on_close()`）で `export_service.shutdown()` を呼ぶ。

7) ログ/例外の扱い
- [ ] 変更: シャットダウン経路内の例外はログ DEBUG レベルで抑制し、既定ではユーザに見せない。必要な場合のみ 1 行の WARN を出す。

8) シグナル/atexit の統一
- [ ] 追加: `src/api/sketch.py` 起動時に `signal` ハンドラと `atexit` の登録を行う（macOS の spawn 方式でも安全な関数だけを参照）。

9) 設定/タイムアウトの外部化
- [ ] 追加: `PXD_SHUTDOWN_TIMEOUT`（秒）を導入。未設定は 1.0。急ぐ環境では 0.2 などに短縮可能。

10) テスト/検証
- [ ] 追加（smoke）: `tests/smoke/test_shutdown_worker_only.py`
  - [ ] `WorkerPool(num_workers=2)` → `tick()` を数回 → `close()` → 子プロセス残存が無いことを確認。
- [ ] 追加（optional/e2e, 手動）: ESC 終了の手動確認手順（下記）。
- [ ] 変更: 上記変更ファイルに対して `ruff/mypy/pytest -q -m smoke` を実施。

## 手動検証手順（最小）
1. `python main.py` を起動。
2. ESC で終了。
3. ターミナルに例外が出ないことを確認。
4. `python -c "import multiprocessing as mp; print(mp.active_children())"` で子プロセス残存が無いことを確認（外部 `ps` でも良い）。
5. 録画（V）や PNG 保存（P）を一度行った後に ESC 終了しても静かに終了することを確認。

## 影響範囲と懸念
- `_WorkerProcess(daemon=False)` への変更により、親が終了するには `close()` 実施が必須となる（現行でも `on_close()` で実施しており非互換は小さい）。
- シグナルハンドラの追加により、他所で独自にハンドラを登録している場合は調整が必要（本プロジェクト内では未使用）。
- ExportService に `shutdown()` を足すが、外部 API として公開していないため破壊的ではない。

## ドキュメント/整合
- [ ] `architecture.md` に「ShutdownCoordinator と終了順序」を追記（`api.sketch`/`engine.runtime.worker` の該当箇所参照付き）。
- [ ] ルート `AGENTS.md` の Build/Test 最小項目に `PXD_SHUTDOWN_TIMEOUT` の注記を追記。

## 確認事項（要回答）
- [ ] ワーカープロセスの `daemon=True` を外して良いか（`join()` を確実に待つ前提にしたい）。
- [ ] デフォルトのシャットダウン待機（`PXD_SHUTDOWN_TIMEOUT`）は 1.0 秒で問題ないか。より短く/長く希望があれば指定ください。
- [ ] OS シグナル（`SIGINT`/`SIGTERM`/`SIGHUP`）でウィンドウがない場合もプロセスを静かに落とす仕様で良いか。

---

実装に着手してよければ、このチェックリストに基づいて進め、完了した項目にチェックを入れていきます。必要に応じて計画の微修正（順序/範囲）もここに追記します。

