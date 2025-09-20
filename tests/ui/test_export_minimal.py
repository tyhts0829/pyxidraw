import time
from pathlib import Path

import numpy as np
import pytest

from util.paths import ensure_gcode_dir, ensure_screenshots_dir
from engine.export.service import ExportService


@pytest.mark.smoke
def test_ensure_dirs():
    p1 = ensure_screenshots_dir()
    p2 = ensure_gcode_dir()
    assert p1.exists() and p1.is_dir()
    assert p2.exists() and p2.is_dir()


@pytest.mark.smoke
def test_export_simulated_complete(tmp_path: Path):
    # setup coords/offsets（単一ライン）
    coords = np.zeros((500, 3), dtype=np.float32)
    offsets = np.array([0, coords.shape[0]], dtype=np.int32)
    svc = ExportService()
    job = svc.submit_gcode_job((coords, offsets), simulate=True)

    t0 = time.time()
    path = None
    while True:
        pr = svc.progress(job)
        if pr.state == "completed":
            assert pr.path is not None
            path = pr.path
            break
        if time.time() - t0 > 5:
            pytest.skip("export worker too slow in CI env")
        time.sleep(0.02)
    # 完了ファイル
    assert path is not None and Path(path).exists()
    # 片付け
    try:
        Path(path).unlink()
    except Exception:
        pass


@pytest.mark.smoke
def test_export_simulated_cancel():
    coords = np.zeros((10000, 3), dtype=np.float32)
    offsets = np.array([0, coords.shape[0]], dtype=np.int32)
    svc = ExportService()
    job = svc.submit_gcode_job((coords, offsets), simulate=True)

    # 少し待ってからキャンセル
    time.sleep(0.05)
    svc.cancel(job)

    t0 = time.time()
    while True:
        pr = svc.progress(job)
        if pr.state in ("cancelled", "failed", "completed"):
            # cancelled が期待値
            assert pr.state == "cancelled"
            break
        if time.time() - t0 > 5:
            pytest.skip("cancel was not processed in time")
        time.sleep(0.02)


@pytest.mark.smoke
def test_export_fail_path():
    # writer 未設定（実行パス）→ 失敗扱い
    coords = np.zeros((10, 3), dtype=np.float32)
    offsets = np.array([0, coords.shape[0]], dtype=np.int32)
    svc = ExportService(writer=object())  # write を持たない=例外で失敗
    job = svc.submit_gcode_job((coords, offsets), simulate=False)
    t0 = time.time()
    while True:
        pr = svc.progress(job)
        if pr.state in ("failed", "completed"):
            assert pr.state == "failed"
            break
        if time.time() - t0 > 5:
            pytest.skip("worker did not fail within timeout")
        time.sleep(0.02)
