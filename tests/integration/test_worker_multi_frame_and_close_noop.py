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


def _base() -> Geometry:
    l = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([l])


def _draw(t: float, cc: Mapping[int, int], *, coords, offsets) -> Geometry:  # pragma: no cover
    g = Geometry(coords.copy(), offsets.copy())
    return g.translate(min(t, 1.0) * 0.01, 0.0, 0.0)


@pytest.mark.integration
def test_multi_frame_delivery_and_tick_after_close_noop():
    base = _base()
    pool = WorkerPool(
        fps=120,
        draw_callback=functools.partial(_draw, coords=base.coords, offsets=base.offsets),
        cc_snapshot=lambda: {},
        num_workers=1,
    )
    swap = SwapBuffer()
    recv = StreamReceiver(swap, pool.result_q, max_packets_per_tick=8)

    # Aim to receive at least two frames quickly
    got = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3 and got < 2:
        pool.tick(0.008)
        recv.tick(0.0)
        if swap.try_swap():
            got += 1
        time.sleep(0.002)

    assert got >= 2, "expected at least two frames delivered"

    # After close, tick() should be a no-op and not raise
    pool.close()
    for _ in range(3):
        pool.tick(0.016)
