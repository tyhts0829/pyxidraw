"""
どこで: `engine.runtime` のワーカ実行層。
何を: `RenderTask` を生成してワーカ（プロセス/インライン）へ渡し、`draw_callback` を実行して
      `Geometry` を得る。結果は `RenderPacket` としてキューへ送出し、例外は `WorkerTaskError`
      で文脈付きに伝搬。`tick(dt)` は FPS に従いタスクを発行し、`close()` は安全に停止する。
なぜ: 生成（CPU 計算）をメインスレッドから切り離し、描画ループをブロックせずに安定駆動するため。

注意（重要）:
- macOS など `multiprocessing` が spawn 方式の環境では、サブプロセスに渡すコールバック
  （`draw_callback`/`apply_cc_snapshot`/`metrics_snapshot`）は「ピクル可能＝トップレベル定義」である必要がある。
  ローカル関数や `functools.partial` でクロージャを含むものは避けること。
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
from queue import Full, Queue
from typing import Callable, Mapping, cast

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry

from ..core.tickable import Tickable
from .packet import RenderPacket
from .task import RenderTask


class WorkerTaskError(Exception):
    """Draw コールバック中の例外をラップしてフレームID等の文脈を付与。

    multiprocessing 経由のシリアライズ/デシリアライズに耐えるよう、
    単一のメッセージ引数でも初期化できるようにする。
    """

    def __init__(
        self, frame_id=None, original: Exception | None = None, message: str | None = None, *args
    ):
        # Unpickle 経路（例外は message だけで復元されることがある）
        if message is None and isinstance(frame_id, str) and original is None:
            message = frame_id
            frame_id = None

        if message is None:
            message = f"WorkerTaskError(frame_id={frame_id}): {original}"
        super().__init__(message)
        self.frame_id = frame_id
        self.original = original

    def __reduce__(self):
        # ピクル化時はメッセージのみで再構築できるようにする
        return (WorkerTaskError, (str(self),))


class _WorkerProcess(mp.Process):
    """バックグラウンドで draw_callback を呼び RenderPacket を生成する。

    ピクル要件:
    - `draw_callback`/`apply_cc_snapshot`/`metrics_snapshot` はトップレベル関数など
      ピクル可能な参照を渡すこと（spawn 互換）。
    """

    def __init__(
        self,
        task_q: mp.Queue,
        result_q: mp.Queue,
        draw_callback: Callable[[float], Geometry],
        apply_cc_snapshot: Callable[[Mapping[int, float] | None], None] | None,
        apply_param_snapshot: Callable[[Mapping[str, object] | None, float], None] | None,
        metrics_snapshot: Callable[[], Mapping[str, Mapping[str, int]]] | None,
    ):
        super().__init__(daemon=True)
        self.task_q, self.result_q = task_q, result_q
        self.draw_callback = draw_callback
        self.apply_cc_snapshot = apply_cc_snapshot
        self.apply_param_snapshot = apply_param_snapshot
        self.metrics_snapshot = metrics_snapshot

    def run(self) -> None:
        """頂点データとフレームIDを持つ RenderPacket を生成し、結果キューに送る。"""
        logger = logging.getLogger(__name__)
        for task in iter(self.task_q.get, None):  # None = sentinel
            try:
                # CC スナップショットを適用（API レイヤから受け取った関数を使用）
                try:
                    if self.apply_cc_snapshot is not None:
                        self.apply_cc_snapshot(task.cc_state)
                except Exception:
                    pass
                # Parameter スナップショットを適用（SnapshotRuntime を有効化）
                try:
                    if self.apply_param_snapshot is not None:
                        self.apply_param_snapshot(getattr(task, "param_overrides", None), task.t)
                except Exception:
                    pass
                # 直前のスナップショット
                try:
                    before = self.metrics_snapshot() if self.metrics_snapshot is not None else None
                except Exception:
                    before = None
                # t のみを渡す（cc は api.cc 側で参照）
                geometry = self.draw_callback(task.t)
                # LazyGeometry はワーカ側で実体化してメインスレッド負荷を避ける
                try:
                    if isinstance(geometry, LazyGeometry):
                        geometry = geometry.realize()
                except Exception:
                    pass
                # Runtime をクリア（スナップショット無しを渡して明示的に無効化）
                try:
                    if self.apply_param_snapshot is not None:
                        self.apply_param_snapshot(None, 0.0)
                except Exception:
                    pass
                # 直後のスナップショット
                try:
                    after = self.metrics_snapshot() if self.metrics_snapshot is not None else None
                except Exception:
                    after = None
                # HIT/MISS の二値判定（MISS 優先: フレーム内で 1 つでも MISS 増分があれば MISS）
                flags = None
                if isinstance(before, dict) and isinstance(after, dict):
                    try:
                        e_hits_after = int(after.get("effect", {}).get("hits", 0))
                        e_hits_before = int(before.get("effect", {}).get("hits", 0))
                        e_miss_after = int(after.get("effect", {}).get("misses", 0))
                        e_miss_before = int(before.get("effect", {}).get("misses", 0))
                        s_hits_after = int(after.get("shape", {}).get("hits", 0))
                        s_hits_before = int(before.get("shape", {}).get("hits", 0))
                        s_miss_after = int(after.get("shape", {}).get("misses", 0))
                        s_miss_before = int(before.get("shape", {}).get("misses", 0))

                        e_miss = e_miss_after > e_miss_before
                        e_hit = e_hits_after > e_hits_before
                        s_miss = s_miss_after > s_miss_before
                        s_hit = s_hits_after > s_hits_before

                        # フレーム差分に基づくフラグ（増分なしのキーは送らない）
                        flags = {}
                        if e_miss:
                            flags["effect"] = "MISS"
                        elif e_hit:
                            flags["effect"] = "HIT"
                        if s_miss:
                            flags["shape"] = "MISS"
                        elif s_hit:
                            flags["shape"] = "HIT"
                        # デバッグ出力は抑制
                    except Exception:
                        flags = None
                self.result_q.put(
                    RenderPacket(geometry=geometry, frame_id=task.frame_id, cache_flags=flags)
                )
            except Exception as e:  # 例外を親へ
                # デバッグ用：分類情報を付与
                import traceback

                logger.exception(
                    "[worker] stage=draw_callback frame_id=%s error=%s\n%s",
                    getattr(task, "frame_id", -1),
                    e,
                    traceback.format_exc(),
                )
                self.result_q.put(WorkerTaskError(getattr(task, "frame_id", -1), e))


class WorkerPool(Tickable):
    """タスク生成とワーカープール管理のみを担当。"""

    def __init__(
        self,
        fps: int,
        draw_callback: Callable[[float], Geometry],
        cc_snapshot,
        num_workers: int = 4,
        apply_cc_snapshot: Callable[[Mapping[int, float] | None], None] | None = None,
        apply_param_snapshot: Callable[[Mapping[str, object] | None, float], None] | None = None,
        param_snapshot: Callable[[], Mapping[str, object] | None] | None = None,
        metrics_snapshot: Callable[[], Mapping[str, Mapping[str, int]]] | None = None,
    ):
        self._fps = fps
        self._frame_iter = itertools.count()
        self._elapsed_time = 0.0
        self._task_q: mp.Queue = mp.Queue(maxsize=2 * num_workers)
        self._result_q: Queue[RenderPacket | WorkerTaskError] | mp.Queue
        self._cc_snapshot = cc_snapshot
        self._param_snapshot = param_snapshot or (lambda: None)
        self._draw_callback = draw_callback
        self._apply_cc_snapshot = apply_cc_snapshot
        self._apply_param_snapshot = apply_param_snapshot
        self._metrics_snapshot = metrics_snapshot
        self._inline = num_workers < 1
        if self._inline:
            # スレッド内で完結させるため、シリアライズを避ける queue.Queue を利用する
            self._result_q = Queue()
            self._workers: list[_WorkerProcess] = []
        else:
            self._result_q = mp.Queue()
            self._workers = [
                _WorkerProcess(
                    self._task_q,
                    self._result_q,
                    draw_callback,
                    apply_cc_snapshot,
                    apply_param_snapshot,
                    metrics_snapshot,
                )
                for _ in range(num_workers)
            ]
            for w in self._workers:
                w.start()
        # 冪等な close() のための内部フラグ
        self._closed: bool = False

    # -------- Tickable interface --------
    def tick(self, dt: float) -> None:  # dt は今回は未使用
        """FPS に従いタスクをキューイング。Queue が詰まっていれば無視。"""
        if getattr(self, "_closed", False):
            return
        self._elapsed_time += dt
        try:
            frame_id = next(self._frame_iter)
            task = RenderTask(
                frame_id=frame_id,
                t=self._elapsed_time,
                cc_state=self._cc_snapshot(),
                param_overrides=self._param_snapshot(),
            )
            if self._inline:
                try:
                    # CC スナップショットを適用（インライン時）
                    try:
                        if self._apply_cc_snapshot is not None:
                            self._apply_cc_snapshot(task.cc_state)
                    except Exception:
                        pass
                    # Parameter スナップショットを適用（インライン時）
                    try:
                        if self._apply_param_snapshot is not None:
                            self._apply_param_snapshot(task.param_overrides, task.t)
                    except Exception:
                        pass
                    # 直前/直後で差分を取得
                    try:
                        before = (
                            self._metrics_snapshot() if self._metrics_snapshot is not None else None
                        )
                    except Exception:
                        before = None
                    geometry = self._draw_callback(task.t)
                    # LazyGeometry はインライン実行でもここで実体化
                    try:
                        if isinstance(geometry, LazyGeometry):
                            geometry = geometry.realize()
                    except Exception:
                        pass
                    # Runtime をクリア
                    try:
                        if self._apply_param_snapshot is not None:
                            self._apply_param_snapshot(None, 0.0)
                    except Exception:
                        pass
                    try:
                        after = (
                            self._metrics_snapshot() if self._metrics_snapshot is not None else None
                        )
                    except Exception:
                        after = None
                    flags = None
                    if isinstance(before, dict) and isinstance(after, dict):
                        try:
                            s_hits_after = int(after.get("shape", {}).get("hits", 0))
                            s_hits_before = int(before.get("shape", {}).get("hits", 0))
                            s_miss_after = int(after.get("shape", {}).get("misses", 0))
                            s_miss_before = int(before.get("shape", {}).get("misses", 0))
                            e_hits_after = int(after.get("effect", {}).get("hits", 0))
                            e_hits_before = int(before.get("effect", {}).get("hits", 0))
                            e_miss_after = int(after.get("effect", {}).get("misses", 0))
                            e_miss_before = int(before.get("effect", {}).get("misses", 0))

                            s_miss = s_miss_after > s_miss_before
                            s_hit = s_hits_after > s_hits_before
                            e_miss = e_miss_after > e_miss_before
                            e_hit = e_hits_after > e_hits_before

                            flags = {}
                            if s_miss:
                                flags["shape"] = "MISS"
                            elif s_hit:
                                flags["shape"] = "HIT"
                            if e_miss:
                                flags["effect"] = "MISS"
                            elif e_hit:
                                flags["effect"] = "HIT"
                            # デバッグ出力は抑制
                        except Exception:
                            flags = None
                    self._result_q.put(
                        RenderPacket(geometry=geometry, frame_id=task.frame_id, cache_flags=flags)
                    )
                except Exception as exc:  # 例外を揃える
                    self._result_q.put(WorkerTaskError(task.frame_id, exc))
            else:
                self._task_q.put_nowait(task)
        except Full:
            pass  # ワーカが追いついていない

    # --------- public API ---------
    @property
    def result_q(self) -> Queue[RenderPacket | WorkerTaskError] | mp.Queue:
        return self._result_q

    def close(self) -> None:
        """ワーカプールを停止してキューをクローズ（多重呼び出しに安全）。"""
        if getattr(self, "_closed", False):
            return
        # 以降の例外で途中離脱しても、次回は no-op になるよう先にフラグを立てる
        self._closed = True
        if self._inline:
            return
        try:
            for _ in self._workers:
                try:
                    self._task_q.put_nowait(None)
                except Exception:
                    pass
            for w in self._workers:
                w.join(timeout=1.0)
                if w.is_alive():
                    w.terminate()
        finally:
            try:
                self._task_q.close()
            except Exception:
                pass
            if not self._inline:
                try:
                    cast(mp.Queue, self._result_q).close()
                except Exception:
                    pass
