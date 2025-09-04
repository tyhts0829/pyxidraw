import pytest

from api.runner import run_sketch
import types


def _draw(_t, _cc):
    # Lazy import to avoid GL unless actually called
    from engine.core.geometry import Geometry
    import numpy as np

    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_runner_converts_invalid_midi_to_systemexit(monkeypatch):
    # Make connect_midi_controllers raise the specific error
    from engine.io.controller import InvalidPortError

    def _raise():
        raise InvalidPortError("no ports")

    monkeypatch.setattr("engine.io.manager.connect_midi_controllers", _raise)
    # Stub pyglet/moderngl to avoid window/context creation during import
    pyglet_mod = types.ModuleType("pyglet")
    pyglet_window_mod = types.ModuleType("pyglet.window")
    pyglet_window_mod.key = types.SimpleNamespace()
    class _Win:
        def __init__(self, *a, **k):
            pass
        def event(self, fn):
            return fn
    pyglet_window_mod.Window = _Win
    class _FPS:
        def __init__(self, *a, **k):
            pass
        def draw(self):
            pass
    pyglet_window_mod.FPSDisplay = _FPS
    pyglet_mod.window = pyglet_window_mod
    pyglet_mod.app = types.SimpleNamespace(run=lambda: None, exit=lambda: None)
    pyglet_mod.clock = types.SimpleNamespace(schedule_interval=lambda *a, **k: None)
    # stub pyglet.gl
    pyglet_gl_mod = types.ModuleType("pyglet.gl")
    class _Cfg:  # noqa: D401
        pass
    pyglet_gl_mod.Config = _Cfg
    pyglet_gl_mod.glClearColor = lambda *a, **k: None
    mgl_stub = types.SimpleNamespace(create_context=lambda: types.SimpleNamespace(enable=lambda *a, **k: None, blend_func=None))
    sysm = __import__("sys").modules
    monkeypatch.setitem(sysm, "pyglet", pyglet_mod)
    monkeypatch.setitem(sysm, "pyglet.window", pyglet_window_mod)
    monkeypatch.setitem(sysm, "pyglet.gl", pyglet_gl_mod)
    monkeypatch.setitem(__import__("sys").modules, "moderngl", mgl_stub)

    with pytest.raises(SystemExit) as ex:
        run_sketch(_draw, use_midi=True, fps=1)
    assert ex.value.code == 2
