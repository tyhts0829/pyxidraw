import io

import numpy as np
import pytest

from engine.export.gcode import GCodeParams, GCodeWriter


def _run_writer(coords: np.ndarray, offsets: np.ndarray, params: GCodeParams) -> list[str]:
    fp = io.StringIO()
    GCodeWriter().write(coords, offsets, params, fp)  # type: ignore[arg-type]
    fp.seek(0)
    return [line.rstrip("\n") for line in fp.getvalue().splitlines()]


@pytest.mark.smoke
def test_minimal_two_lines_outputs_sequences():
    # line0: (0,0)->(10,0), line1: (10,10)->(20,20)
    pts = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [20.0, 20.0, 0.0]], dtype=np.float32
    )
    offsets = np.array([0, 2, 4], dtype=np.int32)
    params = GCodeParams(
        travel_feed=1500,
        draw_feed=1000,
        z_up=3.0,
        z_down=-2.0,
        y_down=False,
        origin=(91.0, -0.75),
        decimals=3,
    )

    lines = _run_writer(pts, offsets, params)

    # Header essentials
    assert any(l.startswith("G21") for l in lines)
    assert any(l.startswith("G90") for l in lines)
    assert any(l.startswith("G28") for l in lines)
    assert any("M420 S1 Z10" in l for l in lines)

    # Body for first line
    i0 = lines.index("; line 0 start")
    assert lines[i0 + 1] == "G1 Z3.0"
    assert lines[i0 + 2] == "G1 F1500"
    # first vertex with origin added
    assert lines[i0 + 3].startswith("G1 X91.0 Y-0.75")
    # switch to draw
    assert lines[i0 + 4] == "G1 Z-2.0"
    assert lines[i0 + 5] == "G1 F1000"
    # second vertex
    assert lines[i0 + 6].startswith("G1 X101.0 Y-0.75")

    # Body for second line (another travel then draw)
    i1 = lines.index("; line 1 start")
    assert lines[i1 + 1] == "G1 Z3.0"
    assert lines[i1 + 2] == "G1 F1500"
    assert lines[i1 + 3].startswith("G1 X101.0 Y9.25")
    assert lines[i1 + 4] == "G1 Z-2.0"
    assert lines[i1 + 5] == "G1 F1000"
    assert lines[i1 + 6].startswith("G1 X111.0 Y19.25")


def test_y_flip_simple():
    pts = np.array([[0.0, 10.0], [10.0, 20.0]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    params = GCodeParams(
        travel_feed=1500,
        draw_feed=1000,
        z_up=3.0,
        z_down=-2.0,
        y_down=True,  # simple inversion y -> -y
        origin=(0.0, 0.0),
        decimals=3,
    )
    lines = _run_writer(pts, offsets, params)
    i0 = lines.index("; line 0 start")
    # first and second vertex Y should be negative (inverted)
    assert lines[i0 + 3].startswith("G1 X0.0 Y-10.0")
    assert lines[i0 + 6].startswith("G1 X10.0 Y-20.0")


def test_y_flip_with_canvas_height():
    # Strict inversion using canvas height: y -> H - y
    pts = np.array([[0.0, 10.0], [10.0, 20.0]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    params = GCodeParams(
        travel_feed=1500,
        draw_feed=1000,
        z_up=3.0,
        z_down=-2.0,
        y_down=True,
        origin=(0.0, 0.0),
        decimals=3,
        canvas_height_mm=100.0,
    )
    lines = _run_writer(pts, offsets, params)
    i0 = lines.index("; line 0 start")
    # y=10 -> 90, y=20 -> 80
    assert lines[i0 + 3].startswith("G1 X0.0 Y90.0")
    assert lines[i0 + 6].startswith("G1 X10.0 Y80.0")


def test_connect_distance_merges_motion():
    # Two lines with second starting at the end of first
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    offsets = np.array([0, 2, 4], dtype=np.int32)
    params = GCodeParams(
        travel_feed=1500,
        draw_feed=1000,
        z_up=3.0,
        z_down=-2.0,
        y_down=False,
        origin=(0.0, 0.0),
        decimals=3,
        connect_distance=0.5,
    )

    lines = _run_writer(pts, offsets, params)

    # Only one travel block should appear when connected
    travel_cmds = [l for l in lines if l == "G1 F1500"]
    assert len(travel_cmds) == 1
    # Count Z-up inside body (exclude footer)
    try:
        footer_idx = lines.index("; ====== Footer ======")
    except ValueError:
        footer_idx = len(lines)
    z_up_cmds_body = [l for l in lines[:footer_idx] if l == "G1 Z3.0"]
    assert len(z_up_cmds_body) == 1


def test_bed_range_violation_raises():
    pts = np.array([[1000.0, 0.0], [1001.0, 0.0]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    params = GCodeParams(
        travel_feed=1500,
        draw_feed=1000,
        z_up=3.0,
        z_down=-2.0,
        y_down=False,
        origin=(0.0, 0.0),
        decimals=3,
        bed_range=(0.0, 300.0),
    )
    with pytest.raises(ValueError):
        _run_writer(pts, offsets, params)
