# インラインワーカーの queue 差し替え計画

## 指摘概要
- [P1] `src/engine/runtime/worker.py:93-102` のインラインワーカーモードで `multiprocessing.Queue` を使用し続けており、全ペイロードの pickling が発生している。

## 詳細
- インライン経路は `num_workers < 1` のときに利用され、マルチプロセス経由のオーバーヘッドを避けることが目的。
- `multiprocessing.Queue` はスレッド間転送でも常にオブジェクトを pickling するため、`RenderPacket` 内の `Geometry` の `numpy.ndarray` など大きなデータがフレームごとにシリアライズされる。
- その結果、インラインモードが O(N) のコピーコストを持つようになり、複雑なスケッチでフォールバック経路が実用にならないほど遅くなるリグレッションが発生している。

## 改善計画
- [x] インラインモードで `multiprocessing.Queue` の代わりに `queue.Queue` を使用するよう復元する。
- [x] `inline` 分岐でのキュー初期化ロジックを整理し、インライン専用でシリアライズを避けるコメントを追加（既存スモークテストでカバレッジ確認）。
- [x] 変更後に関連する単体テスト（`pytest -q tests/runtime/test_worker_pool_smoke.py`）を実行し、`num_workers == 0` ケースの退行が無いことを確認。
- [x] `architecture.md` に該当の記述は無かったため更新不要と判断（差分なし）。

## 追記メモ
- [2025-09-20] インライン経路の型アノテーションに伴う `Queue.close` 未定義警告を、`cast` を用いた `mp.Queue` 明示化で解消。
