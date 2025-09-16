from __future__ import annotations

import time
from typing import Mapping

import numpy as np
import pytest

from engine.core.geometry import Geometry
from engine.runtime.worker import WorkerPool


def _draw_cb(t: float, _cc: Mapping[int, float]) -> Geometry:  # noqa: ANN001
    pts = np.array([[0.0 + t * 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([pts])


@pytest.mark.integration
def test_worker_pool_emits_packets_and_closes_cleanly() -> None:
    pool = WorkerPool(fps=60, draw_callback=_draw_cb, cc_snapshot=lambda: {}, num_workers=1)
    try:
        # 少しだけ tick してタスクを投入
        for _ in range(3):
            pool.tick(1.0 / 60)
            time.sleep(0.02)
        # 受信確認
        got = []
        start = time.time()
        while time.time() - start < 2.0:
            try:
                pkt = pool.result_q.get_nowait()
                if not isinstance(pkt, Exception):
                    got.append(pkt)
                    break
            except Exception:
                time.sleep(0.01)
        assert got, "WorkerPool からパケットを受信できませんでした"
    finally:
        pool.close()
