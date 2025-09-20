import sys
import types


def _ensure_pyglet_stub() -> None:
    if "pyglet" in sys.modules:
        return

    pyglet_stub = types.ModuleType("pyglet")

    class DummyLabel:
        def __init__(
            self, *_, text: str = "", x: float = 0.0, y: float = 0.0, **__kwargs: object
        ) -> None:
            self.text = text
            self.x = x
            self.y = y

    text_module = types.ModuleType("pyglet.text")
    setattr(text_module, "Label", DummyLabel)

    class DummyRectangle:
        def __init__(
            self,
            *,
            x: float = 0.0,
            y: float = 0.0,
            width: float = 0.0,
            height: float = 0.0,
            color: tuple[int, int, int] | None = None,
            batch: object | None = None,
            group: object | None = None,
        ) -> None:
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.color = color or (0, 0, 0)
            self.batch = batch
            self.group = group

    shapes_module = types.ModuleType("pyglet.shapes")
    setattr(shapes_module, "Rectangle", DummyRectangle)

    class DummyBatch:
        def draw(self) -> None:
            return None

    class DummyGroup:
        def __init__(self, *, order: int = 0) -> None:
            self.order = order

    graphics_module = types.ModuleType("pyglet.graphics")
    setattr(graphics_module, "Batch", DummyBatch)
    setattr(graphics_module, "Group", DummyGroup)

    class DummyKey:
        MOD_SHIFT = 1 << 0
        MOD_COMMAND = 1 << 1
        MOD_CTRL = 1 << 2
        MOD_ACCEL = MOD_COMMAND

    window_module = types.ModuleType("pyglet.window")
    setattr(window_module, "key", DummyKey)

    setattr(pyglet_stub, "text", text_module)
    setattr(pyglet_stub, "shapes", shapes_module)
    setattr(pyglet_stub, "graphics", graphics_module)
    setattr(pyglet_stub, "window", window_module)
    setattr(
        pyglet_stub,
        "clock",
        types.SimpleNamespace(schedule_interval=lambda *args, **kwargs: None),
    )

    sys.modules["pyglet"] = pyglet_stub
    sys.modules["pyglet.text"] = text_module
    sys.modules["pyglet.shapes"] = shapes_module
    sys.modules["pyglet.graphics"] = graphics_module
    sys.modules["pyglet.window"] = window_module


_ensure_pyglet_stub()

import pytest

from engine.ui.parameters.normalization import denormalize_scalar
from engine.ui.parameters.panel import SliderWidget
from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    RangeHint,
    ValueType,
)


def _make_store_with_descriptor(
    *, value_type: ValueType, hint: RangeHint, default: float
) -> tuple[ParameterStore, ParameterDescriptor]:
    store = ParameterStore()
    descriptor = ParameterDescriptor(
        id="shape.test#0.value",
        label="test · value",
        source="shape",
        category="shape",
        value_type=value_type,
        default_value=default,
        range_hint=hint,
    )
    store.register(descriptor, default)
    return store, descriptor


def test_slider_widget_drag_respects_mapped_range_for_int():
    hint = RangeHint(
        min_value=0.0,
        max_value=1.0,
        mapped_min=3,
        mapped_max=11,
        mapped_step=1,
    )
    store, descriptor = _make_store_with_descriptor(value_type="int", hint=hint, default=0.5)
    layout = ParameterLayoutConfig()
    slider = SliderWidget(descriptor, store, layout)
    slider.set_bounds(x=0, y=0, width=200, height=20)

    slider.begin_drag()
    slider.drag_to(slider.x + slider.width)

    normalized = store.current_value(descriptor.id)
    assert normalized == pytest.approx(1.0)

    actual = denormalize_scalar(normalized, hint, value_type="int")
    assert actual == 11


def test_slider_widget_actual_value_matches_mapped_range():
    hint = RangeHint(
        min_value=0.0,
        max_value=1.0,
        mapped_min=10.0,
        mapped_max=110.0,
    )
    store, descriptor = _make_store_with_descriptor(value_type="float", hint=hint, default=0.25)
    layout = ParameterLayoutConfig()
    slider = SliderWidget(descriptor, store, layout)
    slider.set_bounds(x=5, y=0, width=100, height=20)

    slider.begin_drag()
    slider.drag_to(slider.x + slider.width * 0.75)

    normalized = store.current_value(descriptor.id)
    assert normalized == pytest.approx(0.75, rel=1e-5)

    actual_expected = denormalize_scalar(normalized, hint, value_type="float")
    assert slider._current_actual() == pytest.approx(actual_expected, rel=1e-6)
