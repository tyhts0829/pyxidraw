import numpy as np

from engine.render.renderer import LineRenderer
from engine.render.types import Layer
from engine.runtime.buffer import SwapBuffer


class DummyProgram(dict):
    def __init__(self) -> None:
        super().__init__()
        self["color"] = type("V", (), {"value": (0.0, 0.0, 0.0, 1.0)})()
        self["projection"] = type("P", (), {"write": lambda self, b: None})()  # type: ignore[misc]


class DummyMesh:
    def __init__(self, ctx, program, primitive_restart_index) -> None:  # noqa: D401, ANN001
        self.ctx = ctx
        self.program = program
        self.primitive_restart_index = primitive_restart_index
        self.index_count = 0
        self.vao = type("V", (), {"render": lambda self, mode, count: None})()  # type: ignore[misc]

    def release(self) -> None:
        return None


class DummyCtx:
    def clear(self, *color):  # type: ignore[no-untyped-def]
        self.last_clear = tuple(color)


class DummySwap(SwapBuffer):
    def __init__(self) -> None:
        self._front = None

    def try_swap(self) -> bool:  # type: ignore[override]
        return False

    def get_front(self):  # type: ignore[override]
        return self._front


def _make_renderer(line_color):
    ctx = DummyCtx()
    proj = np.eye(4, dtype="f4")

    swap = DummySwap()

    # patch LineMesh/Shader to avoid real GL
    import engine.render.renderer as rmod

    orig_line_mesh = rmod.LineMesh
    orig_shader = rmod.Shader

    class DummyShader:
        @staticmethod
        def create_shader(_ctx):
            return DummyProgram()

    try:
        rmod.LineMesh = DummyMesh  # type: ignore[assignment]
        rmod.Shader = DummyShader  # type: ignore[assignment]
        renderer = LineRenderer(ctx, proj, swap, line_color=line_color)
    finally:
        rmod.LineMesh = orig_line_mesh  # type: ignore[assignment]
        rmod.Shader = orig_shader  # type: ignore[assignment]
    return renderer


def test_set_base_line_color_updates_base_and_current():
    renderer = _make_renderer((0.0, 0.0, 0.0, 1.0))
    renderer.set_base_line_color((0.2, 0.4, 0.6, 1.0))
    # _base_line_color と line_program["color"].value が揃っていること
    assert tuple(renderer._base_line_color) == (0.2, 0.4, 0.6, 1.0)  # type: ignore[attr-defined]
    assert tuple(renderer.line_program["color"].value) == (0.2, 0.4, 0.6, 1.0)


def test_layer_without_color_uses_base_line_color(monkeypatch):
    renderer = _make_renderer((0.0, 0.0, 0.0, 1.0))
    renderer.set_base_line_color((0.1, 0.2, 0.3, 1.0))
    # レイヤー1つ、color=None のケースをシミュレート
    renderer._frame_layers = [  # type: ignore[attr-defined]
        Layer(geometry=None, color=None, thickness=None),
    ]
    renderer.draw()
    # draw 後に line_program の色が _base_line_color になっていること
    assert tuple(renderer.line_program["color"].value) == tuple(  # type: ignore[misc]
        renderer._base_line_color  # type: ignore[attr-defined]
    )
