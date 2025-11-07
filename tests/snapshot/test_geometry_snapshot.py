from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from engine.core.geometry import Geometry

SNAP_DIR = Path(__file__).resolve().parents[2] / "tests" / "_snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.snapshot
def test_geometry_snapshot_updateable() -> None:
    """Geometry の coords/offsets スナップショット検証。

    - PXD_UPDATE_SNAPSHOTS=1 のときは現在値で更新
    - それ以外はスナップショットが存在すれば比較、無ければ skip
    """
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    g = Geometry.from_lines([pts])
    snap = {"coords": g.coords.tolist(), "offsets": g.offsets.tolist()}

    snap_path = SNAP_DIR / "geometry_line.json"
    if os.environ.get("PXD_UPDATE_SNAPSHOTS"):
        snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        pytest.skip("snapshot updated")
    if not snap_path.exists():
        pytest.skip("snapshot not found; set PXD_UPDATE_SNAPSHOTS=1 to create")

    data = json.loads(snap_path.read_text(encoding="utf-8"))
    # 旧フォーマット（digest のみ）の場合は更新を促して skip
    if "digest" in data:
        pytest.skip("outdated snapshot format; set PXD_UPDATE_SNAPSHOTS=1 to update")
    assert data.get("coords") == snap["coords"]
    assert data.get("offsets") == snap["offsets"]
