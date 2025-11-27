"""
どこで: `engine.runtime` のワーカ実行層。
何を: `RenderTask` を生成してワーカ（プロセス/インライン）へ渡し、`draw_callback` を実行して
      `Geometry` を得る。結果は `RenderPacket` としてキューへ送出し、例外は `WorkerTaskError`
      で文脈付きに伝搬。`tick(dt)` は呼び出し側フレームクロックから渡される dt を積算して
      `RenderTask` を発行し、`close()` は安全に停止する。
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

import logging
import multiprocessing as mp
from queue import Full, Queue
from typing import Callable, Mapping, Sequence, cast

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import Layer
from util.color import normalize_color

from ..core.tickable import Tickable
from .packet import RenderPacket
from .task import RenderTask


class WorkerTaskError(Exception):
    """Draw コールバック中の例外をラップしてフレームID等の文脈を付与。

    multiprocessing 経由のシリアライズ/デシリアライズに耐えるよう、
    単一のメッセージ引数でも初期化できるようにする。
    """

    def __init__(
        self,
        frame_id: int | None = None,
        original: Exception | None = None,
        message: str | None = None,
    ) -> None:
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


def _normalize_rgba(value: object | None) -> tuple[float, float, float, float] | None:
    """色指定を RGBA(0–1) へ正規化する。

    None はそのまま None を返し、異常な値は util.color.normalize_color に任せて例外として扱う。
    """
    if value is None:
        return None
    r, g, b, a = normalize_color(value)
    return float(r), float(g), float(b), float(a)


def _normalize_to_layers(
    result: Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer],
) -> tuple[Geometry | LazyGeometry | None, list[Layer] | None]:
    """draw() の戻り値を RenderPacket 用に正規化（レイヤー化）する。

    - Sequence が返れば常にレイヤー化。
    - Geometry/LazyGeometry の単体はそのまま geometry として返す。
    """

    def _layer_from(
        g: Geometry | LazyGeometry,
        color: object | None,
        thickness,
        name: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> Layer:
        return Layer(
            geometry=g,
            color=_normalize_rgba(color),
            thickness=float(thickness) if thickness is not None else None,
            name=name,
            meta=meta,
        )

    # 1) Sequence → レイヤー
    if isinstance(result, Sequence) and not isinstance(result, (Geometry, LazyGeometry, Layer)):
        layers: list[Layer] = []
        for item in result:
            if isinstance(item, Layer):
                layers.append(
                    _layer_from(
                        item.geometry,
                        item.color,
                        item.thickness,
                        name=getattr(item, "name", None),
                        meta=getattr(item, "meta", None),
                    )
                )
                continue
            if isinstance(item, LazyGeometry):
                layers.append(_layer_from(item, None, None))
            else:
                # Geometry
                layers.append(_layer_from(item, None, None))
        return None, layers

    # 2) 単体: LazyGeometry に style があれば 1 レイヤー化、なければ geometry
    if isinstance(result, Layer):
        return None, [
            _layer_from(
                result.geometry,
                result.color,
                result.thickness,
                name=getattr(result, "name", None),
                meta=getattr(result, "meta", None),
            )
        ]

    if isinstance(result, LazyGeometry):
        return result, None

    # 3) 単体 Geometry
    return cast(Geometry | LazyGeometry, result), None


def _apply_layer_overrides(
    layers: list[Layer] | None, overrides: Mapping[str, object] | None
) -> list[Layer] | None:
    """layer.* の override をレイヤーへ適用する。"""
    if not layers or not overrides:
        return layers

    # 名前の重複を避けたキーを生成
    keys: list[str] = []
    used: set[str] = set()
    for idx, layer in enumerate(layers):
        base = f"layer{idx}"
        try:
            nm = getattr(layer, "name", None)
            if isinstance(nm, str) and nm:
                base = nm
        except Exception:
            pass
        key = base
        if key in used:
            key = f"{base}_{idx}"
        used.add(key)
        keys.append(key)

    applied: list[Layer] = []
    for layer, key in zip(layers, keys):
        col_pid = f"layer.{key}.color"
        th_pid = f"layer.{key}.thickness"
        color_override = overrides.get(col_pid) if isinstance(overrides, Mapping) else None
        thickness_override = overrides.get(th_pid) if isinstance(overrides, Mapping) else None

        color_final = layer.color
        if color_override is not None:
            color_final = _normalize_rgba(color_override)
        thickness_final = layer.thickness
        if thickness_override is not None:
            try:
                thickness_final = float(thickness_override)
            except Exception:
                pass
        applied.append(
            Layer(
                geometry=layer.geometry,
                color=color_final,
                thickness=thickness_final,
                name=getattr(layer, "name", None),
                meta=getattr(layer, "meta", None),
            )
        )
    return applied


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
            [float],
            Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer],
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
    """タスク生成とワーカープール管理のみを担当。

    時間軸（dt/t）の管理は呼び出し側のフレームクロックに委ね、`tick(dt)` が呼ばれた回数と
    経過時間の累積から `RenderTask(t, frame_id, ...)` を生成してキューへ送出する。
    """

    def __init__(
        self,
        draw_callback: Callable[
            [float],
            Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer],
        ],
        cc_snapshot: Callable[[], Mapping[int, float] | None],
        num_workers: int = 4,
        apply_cc_snapshot: Callable[[Mapping[int, float] | None], None] | None = None,
        apply_param_snapshot: Callable[[Mapping[str, object] | None, float], None] | None = None,
        param_snapshot: Callable[[], Mapping[str, object] | None] | None = None,
        metrics_snapshot: Callable[[], Mapping[str, Mapping[str, int]]] | None = None,
    ):
        self._frame_id: int = 0
        self._elapsed_time = 0.0
        self._task_q: mp.Queue = mp.Queue(maxsize=2 * num_workers)
        self._result_q: Queue[RenderPacket | WorkerTaskError] | mp.Queue
        self._cc_snapshot: Callable[[], Mapping[int, float] | None] = cc_snapshot
        self._param_snapshot: Callable[[], Mapping[str, object] | None] = param_snapshot or (
            lambda: None
        )
        self._draw_callback: Callable[
            [float],
            Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer],
        ] = draw_callback
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
    def tick(self, dt: float) -> None:
        """経過時間 dt を積算し、1 フレーム分の RenderTask をキューイングする。

        Queue が詰まっていれば新しいタスク投入はスキップする。
        """
        if getattr(self, "_closed", False):
            return
        self._elapsed_time += dt
        try:
            task = RenderTask(
                frame_id=self._frame_id,
                t=self._elapsed_time,
                cc_state=self._cc_snapshot(),
                param_overrides=self._param_snapshot(),
            )
            self._frame_id += 1
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


def _apply_cc_and_params_snapshots(
    *,
    t: float,
    cc_state: Mapping[int, float] | None,
    param_overrides: Mapping[str, object] | None,
    apply_cc_snapshot: Callable[[Mapping[int, float] | None], None] | None,
    apply_param_snapshot: Callable[[Mapping[str, object] | None, float], None] | None,
    logger: logging.Logger,
) -> None:
    """CC/パラメータのスナップショットをランタイムに適用する。"""
    if apply_cc_snapshot is not None:
        try:
            apply_cc_snapshot(cc_state)
        except Exception as exc:
            logger.debug("failed to apply CC snapshot: %s", exc, exc_info=True)
    if apply_param_snapshot is not None:
        try:
            apply_param_snapshot(param_overrides, t)
        except Exception as exc:
            logger.debug("failed to apply parameter snapshot: %s", exc, exc_info=True)


def _clear_param_runtime(
    apply_param_snapshot: Callable[[Mapping[str, object] | None, float], None] | None,
    logger: logging.Logger,
) -> None:
    """パラメータ SnapshotRuntime をクリアする。"""
    if apply_param_snapshot is None:
        return
    try:
        apply_param_snapshot(None, 0.0)
    except Exception as exc:
        logger.debug("failed to clear parameter snapshot: %s", exc, exc_info=True)


def _safe_metrics_snapshot(
    metrics_snapshot: Callable[[], Mapping[str, Mapping[str, int]]] | None,
    logger: logging.Logger,
) -> Mapping[str, Mapping[str, int]] | None:
    """メトリクススナップショット取得のラッパ。失敗時はログを出して None を返す。"""
    if metrics_snapshot is None:
        return None
    try:
        return metrics_snapshot()
    except Exception as exc:
        logger.debug("metrics_snapshot failed: %s", exc, exc_info=True)
        return None


def _compute_cache_flags(
    before: Mapping[str, Mapping[str, int]] | None,
    after: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, str] | None:
    """キャッシュメトリクスの差分から HIT/MISS フラグを算出する。"""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
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

        flags: dict[str, str] = {}
        if e_miss:
            flags["effect"] = "MISS"
        elif e_hit:
            flags["effect"] = "HIT"
        if s_miss:
            flags["shape"] = "MISS"
        elif s_hit:
            flags["shape"] = "HIT"
        return flags or None
    except Exception:
        return None


def _execute_draw_to_packet(
    *,
    t: float,
    frame_id: int,
    draw_callback: Callable[
        [float],
        Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer],
    ],
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
        # CC/Parameter スナップショットを適用（SnapshotRuntime を有効化）
        _apply_cc_and_params_snapshots(
            t=t,
            cc_state=cc_state,
            param_overrides=param_overrides,
            apply_cc_snapshot=apply_cc_snapshot,
            apply_param_snapshot=apply_param_snapshot,
            logger=logger,
        )

        # 直前メトリクス
        before = _safe_metrics_snapshot(metrics_snapshot, logger)

        # user draw 実行（t のみ）
        geometry_or_seq = draw_callback(t)
        packet_geom, packet_layers = _normalize_to_layers(geometry_or_seq)
        if packet_layers is not None:
            packet_layers = _apply_layer_overrides(packet_layers, param_overrides)

        # Runtime のクリア（パラメータ）
        _clear_param_runtime(apply_param_snapshot, logger)

        # 直後メトリクス
        after = _safe_metrics_snapshot(metrics_snapshot, logger)

        # HIT/MISS の二値判定（MISS 優先）
        flags = _compute_cache_flags(before, after)

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
