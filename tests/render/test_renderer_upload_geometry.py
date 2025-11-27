from __future__ import annotations

import logging

import numpy as np

from engine.core.geometry import Geometry
from engine.render.renderer import LineRenderer
from util.constants import PRIMITIVE_RESTART_INDEX


class _DummyVAO:
    def __init__(self) -> None:
        self.render_calls: list[tuple[int, int]] = []

    def render(self, mode: int, count: int) -> None:
        self.render_calls.append((mode, count))


class _DummyGpu:
    def __init__(self) -> None:
        self.primitive_restart_index = PRIMITIVE_RESTART_INDEX
        self.index_count = 0
        self.upload_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.update_calls: list[np.ndarray] = []
        self.vao = _DummyVAO()

    def upload(self, verts: np.ndarray, inds: np.ndarray) -> None:
        self.index_count = int(len(inds))
        self.upload_calls.append((verts.copy(), inds.copy()))

    def update_vertices_only(self, verts: np.ndarray) -> None:
        self.update_calls.append(verts.copy())


def _make_geometry() -> Geometry:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([a, b])


def _make_renderer_for_upload() -> tuple[LineRenderer, _DummyGpu]:
    # __init__ を通さず、テストに必要な属性だけを手動で設定する
    renderer = LineRenderer.__new__(LineRenderer)
    renderer._logger = logging.getLogger("test_line_renderer")  # type: ignore[attr-defined]
    renderer._ibo_freeze_enabled = False  # type: ignore[attr-defined]
    renderer._ibo_debug = False  # type: ignore[attr-defined]
    renderer._ibo_uploaded = 0  # type: ignore[attr-defined]
    renderer._ibo_reused = 0  # type: ignore[attr-defined]
    renderer._indices_built = 0  # type: ignore[attr-defined]
    renderer._last_vertex_count = 0  # type: ignore[attr-defined]
    renderer._last_line_count = 0  # type: ignore[attr-defined]
    dummy_gpu = _DummyGpu()
    renderer.gpu = dummy_gpu  # type: ignore[attr-defined]
    return renderer, dummy_gpu


def test_upload_geometry_updates_counts_and_gpu() -> None:
    renderer, gpu = _make_renderer_for_upload()
    geom = _make_geometry()

    renderer._upload_geometry(geom)  # type: ignore[arg-type]

    # GPU にアップロードされ、index_count が設定される
    assert len(gpu.upload_calls) == 1
    assert gpu.index_count > 0
    # HUD 用カウンタも Geometry から計算される
    assert renderer._last_vertex_count == len(geom.coords)  # type: ignore[attr-defined]
    assert renderer._last_line_count == max(0, len(geom.offsets) - 1)  # type: ignore[attr-defined]


def test_upload_geometry_uses_ibo_freeze_on_second_call() -> None:
    renderer, gpu = _make_renderer_for_upload()
    geom = _make_geometry()

    # offsets 署名を事前に設定しておき、freeze 有効で VBO のみ更新経路を通ることを確認する
    offsets = geom.offsets
    off_bytes = offsets.view(np.uint8)
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(off_bytes.tobytes())
    renderer._last_offsets_sig = h.digest()  # type: ignore[attr-defined]

    upload_calls_before = len(gpu.upload_calls)
    first_index_count = gpu.index_count

    # freeze 有効で IBO 固定化経路を通る
    renderer._ibo_freeze_enabled = True  # type: ignore[attr-defined]
    renderer._upload_geometry(geom)  # type: ignore[arg-type]
    assert len(gpu.update_calls) == 1
    assert len(gpu.upload_calls) == upload_calls_before
    # IBO は再利用されるため index_count は変わらない
    assert gpu.index_count == first_index_count
