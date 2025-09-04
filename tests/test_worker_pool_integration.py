import time

from engine.core.geometry import Geometry
from engine.pipeline.worker import WorkerPool


def _draw_cb(t, _cc):
    # minimal geometry
    import numpy as np

    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_worker_pool_minimal_integration():
    # Create a dummy CC snapshot supplier
    def cc_snapshot():
        return {}

    pool = WorkerPool(fps=30, draw_callback=_draw_cb, cc_snapshot=cc_snapshot, num_workers=1)
    try:
        # queue a few frames
        for _ in range(3):
            pool.tick(1 / 60)
            time.sleep(0.02)

        # pull a few results with small wait loop
        q = pool.result_q
        got = 0
        deadline = time.time() + 0.5
        while time.time() < deadline and got == 0:
            while not q.empty():
                pkt = q.get_nowait()
                assert hasattr(pkt, "geometry")
                got += 1
            if got == 0:
                time.sleep(0.01)
        assert got >= 1
    finally:
        pool.close()
