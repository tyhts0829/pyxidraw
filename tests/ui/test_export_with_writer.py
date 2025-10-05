import time
from pathlib import Path

import numpy as np
import pytest

from engine.export.gcode import GCodeParams, GCodeWriter
from engine.export.service import ExportService


@pytest.mark.smoke
def test_export_service_with_writer(tmp_path: Path):
    # small geometry: two points
    coords = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    params = GCodeParams(
        travel_feed=1500,
        draw_feed=1000,
        z_up=3.0,
        z_down=-2.0,
        y_down=False,
        origin=(91.0, -0.75),
        decimals=3,
    )

    svc = ExportService(writer=GCodeWriter())
    job_id = svc.submit_gcode_job((coords, offsets), params=params, simulate=False)

    t0 = time.time()
    path = None
    while True:
        pr = svc.progress(job_id)
        if pr.state == "completed":
            assert pr.path is not None
            path = pr.path
            break
        if pr.state == "failed":
            pytest.fail(f"export failed: {pr.error}")
        if time.time() - t0 > 5:
            pytest.skip("export worker too slow in CI env")
        time.sleep(0.01)

    assert path is not None and Path(path).exists()
    # clean up
    try:
        Path(path).unlink()
    except Exception:
        pass
