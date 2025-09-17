import itertools
import logging
import multiprocessing as mp
from queue import Full, Queue
from typing import Callable, Mapping

from engine.core.geometry import Geometry

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
    """バックグラウンドで draw_callback を呼び RenderPacket を生成する。"""

    def __init__(
        self,
        task_q: mp.Queue,
        result_q: mp.Queue,
        draw_callback: Callable[[float, Mapping[int, float]], Geometry],
    ):
        super().__init__(daemon=True)
        self.task_q, self.result_q = task_q, result_q
        self.draw_callback = draw_callback

    def run(self) -> None:
        """頂点データとフレームIDを持つ RenderPacket を生成し、結果キューに送る。"""
        logger = logging.getLogger(__name__)
        for task in iter(self.task_q.get, None):  # None = sentinel
            try:
                geometry = self.draw_callback(task.t, task.cc_state)
                self.result_q.put(RenderPacket(geometry, task.frame_id))
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
        draw_callback: Callable[[float, Mapping[int, float]], Geometry],
        cc_snapshot,
        num_workers: int = 4,
    ):
        self._fps = fps
        self._frame_iter = itertools.count()
        self._elapsed_time = 0.0
        self._task_q: mp.Queue = mp.Queue(maxsize=2 * num_workers)
        self._result_q: mp.Queue = mp.Queue()
        self._cc_snapshot = cc_snapshot
        self._draw_callback = draw_callback
        self._inline = num_workers < 1
        if self._inline:
            self._workers: list[_WorkerProcess] = []
            self._result_q = Queue()
        else:
            self._result_q = mp.Queue()
            self._workers = [
                _WorkerProcess(self._task_q, self._result_q, draw_callback)
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
            task = RenderTask(frame_id=frame_id, t=self._elapsed_time, cc_state=self._cc_snapshot())
            if self._inline:
                try:
                    geometry = self._draw_callback(task.t, task.cc_state)
                    self._result_q.put(RenderPacket(geometry, task.frame_id))
                except Exception as exc:  # 例外を揃える
                    self._result_q.put(WorkerTaskError(task.frame_id, exc))
            else:
                self._task_q.put_nowait(task)
        except Full:
            pass  # ワーカが追いついていない

    # --------- public API ---------
    @property
    def result_q(self) -> mp.Queue:
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
            try:
                self._result_q.close()
            except Exception:
                pass
