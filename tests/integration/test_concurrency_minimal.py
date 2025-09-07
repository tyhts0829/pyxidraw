from __future__ import annotations

import functools
import time
from typing import Mapping

import numpy as np
import pytest

from engine.core.geometry import Geometry
from engine.pipeline.buffer import SwapBuffer
from engine.pipeline.receiver import StreamReceiver
from engine.pipeline.worker import WorkerPool

# What this tests (TEST_HARDENING_PLAN.md §Integration/Concurrency)
# - Minimal WorkerPool → result_q → StreamReceiver → SwapBuffer path delivers Geometry quickly.
# - WorkerPool.close() is idempotent and returns quickly (guard against leaks in CI).


def _tiny_base_geom() -> Geometry:
    # Two short polylines, extremely cheap to construct
    l1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # 2D -> Z=0
    l2 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([l1, l2])


def draw_cb_translate_x(
    t: float,
    cc: Mapping[int, int],
    *,
    base_coords: np.ndarray,
    base_offsets: np.ndarray,
) -> Geometry:  # pragma: no cover - exercised via integration
    """Top-level function so it's picklable for multiprocessing.

    Translates a tiny base geometry slightly along X based on time.
    """
    dx = float(min(max(t, 0.0), 1.0)) * 0.01
    g = Geometry(base_coords.copy(), base_offsets.copy())
    return g.translate(dx, 0.0, 0.0)


@pytest.mark.integration
def test_worker_receiver_minimal_path() -> None:
    """Minimal end-to-end path: WorkerPool -> result_q -> StreamReceiver -> SwapBuffer.

    Ensures that at least one Geometry arrives, buffer swaps, and version increments quickly.
    """

    base = _tiny_base_geom()
    draw_cb = functools.partial(
        draw_cb_translate_x, base_coords=base.coords, base_offsets=base.offsets
    )

    # cc snapshot supplier: minimal immutable mapping
    def cc_snapshot() -> Mapping[int, int]:
        return {}

    pool = WorkerPool(fps=60, draw_callback=draw_cb, cc_snapshot=cc_snapshot, num_workers=1)
    swap = SwapBuffer()
    recv = StreamReceiver(swap, pool.result_q, max_packets_per_tick=4)

    t0 = time.perf_counter()
    got = False
    # Drive a short tick loop with tiny sleeps to allow the worker process to run
    while time.perf_counter() - t0 < 0.25:  # keep very short (<0.25s)
        pool.tick(0.016)  # queue at most one task per iteration
        recv.tick(0.0)
        if swap.try_swap():
            g = swap.get_front()
            assert isinstance(g, Geometry)
            assert not g.is_empty
            got = True
            break
        time.sleep(0.005)

    try:
        assert got, "No geometry received via minimal path in time budget"
        assert swap.version() >= 1
    finally:
        # Always close to clean up background processes
        pool.close()


@pytest.mark.integration
def test_workerpool_close_idempotent_and_fast() -> None:
    """close() should be multi-call safe and return quickly (<0.2s total here).

    This asserts the stop signal path and idempotency to avoid resource leaks in tests/CI.
    """

    base = _tiny_base_geom()
    draw_cb = functools.partial(
        draw_cb_translate_x, base_coords=base.coords, base_offsets=base.offsets
    )
    pool = WorkerPool(fps=30, draw_callback=draw_cb, cc_snapshot=lambda: {}, num_workers=1)

    # Don't saturate the queue to keep shutdown fast
    pool.tick(0.0)

    t0 = time.perf_counter()
    pool.close()
    # Second call must be a no-op and not raise
    pool.close()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2, f"WorkerPool.close too slow: {elapsed:.3f}s"
