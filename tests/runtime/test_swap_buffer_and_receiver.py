from __future__ import annotations

from queue import Queue

import numpy as np
import pytest

from engine.core.geometry import Geometry
from engine.runtime.buffer import SwapBuffer
from engine.runtime.packet import RenderPacket
from engine.runtime.receiver import StreamReceiver


def test_swap_buffer_push_and_swap() -> None:
    buf = SwapBuffer()
    assert not buf.is_data_ready()
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])
    buf.push(g)
    assert buf.is_data_ready()
    assert buf.try_swap() is True
    assert buf.get_front() is g
    assert buf.version() == 1


def test_stream_receiver_latest_wins_and_exception_is_raised() -> None:
    q: Queue = Queue()
    buf = SwapBuffer()
    r = StreamReceiver(buf, q, max_packets_per_tick=10)

    g0 = Geometry.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])
    g1 = Geometry.from_lines([np.array([[1.0, 0.0, 0.0]], dtype=np.float32)])
    q.put(RenderPacket(g0, frame_id=0))
    q.put(RenderPacket(g1, frame_id=1))  # より新しい
    r.tick(0.0)
    assert buf.try_swap() is True
    assert buf.get_front() is g1

    # 例外パケットはそのまま伝播
    q.put(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        r.tick(0.0)
