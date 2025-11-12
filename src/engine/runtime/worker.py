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

キャッシュに関する注記:
- engine.core の LRU（shape/prefix）はプロセスローカルであり、子プロセス起動時は空の状態から開始する。
- 親プロセスのキャッシュは共有しない。必要に応じて `os.register_at_fork` により子プロセス初期化時の
  キャッシュクリアを登録可能だが、現状は明示の初期化は行っていない（設計上の簡素化）。
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
from queue import Full, Queue
from typing import Callable, Mapping, Sequence, cast

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import StyledLayer

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


# ---- internal helpers -----------------------------------------------------


def _is_style_impl(impl: object) -> bool:
    try:
        kind = getattr(impl, "__effect_kind__", None)
        if kind == "style":
            return True
        name = getattr(impl, "__name__", "")
        return name == "style"
    except Exception:
        return False


def _normalize_to_layers(
    result: Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry],
) -> tuple[Geometry | LazyGeometry | None, list[StyledLayer] | None]:
    """draw() の戻り値を RenderPacket 用に正規化（レイヤー化）する。

    - Sequence が返れば常にレイヤー化。
    - LazyGeometry に style ステップが含まれる場合はレイヤー化し、style ステップは plan から除去。
    - Geometry の単体はそのまま geometry として返す。
    """
    try:
        from util.color import normalize_color as _norm_color
    except Exception:

        def _norm_color(v):  # type: ignore
            return (0.0, 0.0, 0.0, 1.0)

    _DEBUG = False  # renderer 側で集約的に出力するため、ここでは抑制

    # 1) Sequence → レイヤー
    if isinstance(result, Sequence) and not isinstance(result, (Geometry, LazyGeometry)):
        layers: list[StyledLayer] = []
        for item in result:
            if isinstance(item, LazyGeometry):
                # style を抽出して plan から除去
                last_color = None
                last_thickness = None
                filtered: list[tuple[Callable[[Geometry], Geometry], dict[str, object]]] = []
                for impl, params in item.plan:
                    if _is_style_impl(impl):
                        try:
                            c = params.get("color")  # type: ignore[assignment]
                            if isinstance(c, (int, float)):
                                last_color = (float(c), float(c), float(c))
                            else:
                                last_color = c if c is not None else last_color
                            last_thickness = (
                                float(params.get("thickness", 1.0))  # type: ignore[arg-type]
                                if params is not None
                                else last_thickness
                            )
                        except Exception:
                            pass
                        continue
                    filtered.append((impl, dict(params)))
                lg = LazyGeometry(item.base_kind, item.base_payload, filtered)
                rgba = _norm_color(tuple(last_color)) if last_color is not None else None
                layers.append(StyledLayer(geometry=lg, color=rgba, thickness=last_thickness))
            else:
                # Geometry
                layers.append(StyledLayer(geometry=item, color=None, thickness=None))
        return None, layers

    # 2) 単体: LazyGeometry に style があれば 1 レイヤー化、なければ geometry
    if isinstance(result, LazyGeometry):
        last_color = None
        last_thickness = None
        filtered2: list[tuple[Callable[[Geometry], Geometry], dict[str, object]]] = []
        for impl, params in result.plan:
            if _is_style_impl(impl):
                try:
                    c = params.get("color")  # type: ignore[assignment]
                    if isinstance(c, (int, float)):
                        last_color = (float(c), float(c), float(c))
                    else:
                        last_color = c if c is not None else last_color
                    last_thickness = (
                        float(params.get("thickness", 1.0))  # type: ignore[arg-type]
                        if params is not None
                        else last_thickness
                    )
                except Exception:
                    pass
                continue
            filtered2.append((impl, dict(params)))
        if last_color is not None or last_thickness is not None:
            lg = LazyGeometry(result.base_kind, result.base_payload, filtered2)
            rgba = _norm_color(tuple(last_color)) if last_color is not None else None
            return None, [StyledLayer(geometry=lg, color=rgba, thickness=last_thickness)]
        return result, None

    # 3) 単体 Geometry
    return cast(Geometry | LazyGeometry, result), None


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
        draw_callback: Callable[
            [float], Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]
        ],
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
        for task in iter(self.task_q.get, None):  # None = sentinel
            packet, err = _execute_draw_to_packet(
                t=task.t,
                frame_id=task.frame_id,
                draw_callback=self.draw_callback,
                apply_cc_snapshot=self.apply_cc_snapshot,
                apply_param_snapshot=self.apply_param_snapshot,
                cc_state=getattr(task, "cc_state", None),
                param_overrides=getattr(task, "param_overrides", None),
                metrics_snapshot=self.metrics_snapshot,
            )
            if err is not None:
                self.result_q.put(err)
            elif packet is not None:
                self.result_q.put(packet)


class WorkerPool(Tickable):
    """タスク生成とワーカープール管理のみを担当。"""

    def __init__(
        self,
        fps: int,
        draw_callback: Callable[
            [float], Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]
        ],
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
                packet, err = _execute_draw_to_packet(
                    t=task.t,
                    frame_id=task.frame_id,
                    draw_callback=self._draw_callback,
                    apply_cc_snapshot=self._apply_cc_snapshot,
                    apply_param_snapshot=self._apply_param_snapshot,
                    cc_state=task.cc_state,
                    param_overrides=task.param_overrides,
                    metrics_snapshot=self._metrics_snapshot,
                )
                if err is not None:
                    self._result_q.put(err)
                elif packet is not None:
                    self._result_q.put(packet)
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


# ---- 共通化ヘルパ ---------------------------------------------------------


def _execute_draw_to_packet(
    *,
    t: float,
    frame_id: int,
    draw_callback: Callable[[float], Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]],
    apply_cc_snapshot: Callable[[Mapping[int, float] | None], None] | None,
    apply_param_snapshot: Callable[[Mapping[str, object] | None, float], None] | None,
    cc_state: Mapping[int, float] | None,
    param_overrides: Mapping[str, object] | None,
    metrics_snapshot: Callable[[], Mapping[str, Mapping[str, int]]] | None,
) -> tuple[RenderPacket | None, WorkerTaskError | None]:
    """1 フレーム分の draw 実行～正規化～メトリクス差分～Packet 生成を共通化。

    例外は内部で捕捉して `WorkerTaskError` を返し、呼び出し側では put するだけにする。
    """
    logger = logging.getLogger(__name__)
    try:
        # CC スナップショットを適用（存在時のみ）
        try:
            if apply_cc_snapshot is not None:
                apply_cc_snapshot(cc_state)
        except Exception:
            pass
        # Parameter スナップショットを適用（SnapshotRuntime を有効化）
        try:
            if apply_param_snapshot is not None:
                apply_param_snapshot(param_overrides, t)
        except Exception:
            pass

        # 直前メトリクス
        try:
            before = metrics_snapshot() if metrics_snapshot is not None else None
        except Exception:
            before = None

        # user draw 実行（t のみ）
        geometry_or_seq = draw_callback(t)
        packet_geom, packet_layers = _normalize_to_layers(geometry_or_seq)

        # Runtime のクリア（パラメータ）
        try:
            if apply_param_snapshot is not None:
                apply_param_snapshot(None, 0.0)
        except Exception:
            pass

        # 直後メトリクス
        try:
            after = metrics_snapshot() if metrics_snapshot is not None else None
        except Exception:
            after = None

        # HIT/MISS の二値判定（MISS 優先）
        flags: dict[str, str] | None = None
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

                flags = {}
                if e_miss:
                    flags["effect"] = "MISS"
                elif e_hit:
                    flags["effect"] = "HIT"
                if s_miss:
                    flags["shape"] = "MISS"
                elif s_hit:
                    flags["shape"] = "HIT"
            except Exception:
                flags = None

        packet = RenderPacket(
            geometry=packet_geom,
            frame_id=frame_id,
            cache_flags=flags,
            layers=tuple(packet_layers) if packet_layers else None,
        )
        return packet, None
    except Exception as e:
        # 例外を統一ログ（stacktrace 付き）
        logger.exception("[worker] stage=draw_pipeline frame_id=%s error=%s", frame_id, e)
        return None, WorkerTaskError(frame_id, e)
