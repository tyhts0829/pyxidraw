"""Parameter GUI のウィジェット群。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import pyglet
import pyglet.shapes
from pyglet.graphics import Batch, Group
from pyglet.window import key

from .state import ParameterDescriptor, ParameterLayoutConfig, ParameterStore

# スクロール量（ピクセル）
_SCROLL_STEP = 40


class ParameterWidget(Protocol):
    descriptor: ParameterDescriptor

    def set_bounds(self, *, x: float, y: float, width: float, height: float) -> None: ...

    def draw(self, batch: Batch) -> None: ...

    def hit_test(self, x: float, y: float) -> bool: ...

    def begin_drag(self) -> None: ...

    def drag_to(self, x: float, *, modifiers: int = 0) -> None: ...

    def end_drag(self) -> None: ...

    def reset(self) -> None: ...


@dataclass
class SliderWidget:
    """単一パラメータを表す水平スライダー。"""

    descriptor: ParameterDescriptor
    store: ParameterStore
    layout: ParameterLayoutConfig
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    dragging: bool = False

    def __post_init__(self) -> None:
        self._batch: Batch | None = None
        self._label: pyglet.text.Label | None = None
        self._value_label: pyglet.text.Label | None = None
        self._track: pyglet.shapes.Rectangle | None = None
        self._fill: pyglet.shapes.Rectangle | None = None
        self._dirty_bounds = True
        self._track_group = Group(order=0)
        self._fill_group = Group(order=1)
        self._label_group = Group(order=2)
        self._value_group = Group(order=3)
        self._drag_origin_x = 0.0
        self._drag_start_normalized = 0.0

    def set_bounds(self, *, x: float, y: float, width: float, height: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self._dirty_bounds = True

    def _range(self) -> tuple[float, float]:
        hint = self.descriptor.range_hint
        if hint is None:
            return 0.0, 1.0
        return float(hint.min_value), float(hint.max_value)

    def current_value(self) -> Any:
        value = self.store.current_value(self.descriptor.id)
        if value is None:
            return self.descriptor.default_value
        return value

    def _normalized(self) -> float:
        value = self.current_value()
        if not isinstance(value, (int, float)):
            return 0.0
        lo, hi = self._range()
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))

    def begin_drag(self) -> None:
        self.dragging = True
        self._drag_start_normalized = self._normalized()
        self._drag_origin_x = self.x + self._drag_start_normalized * max(self.width, 1.0)

    def end_drag(self) -> None:
        self.dragging = False

    def drag_to(self, x: float, *, modifiers: int = 0) -> None:
        lo, hi = self._range()
        if hi == lo:
            return
        width = max(self.width, 1.0)
        delta = (x - self._drag_origin_x) / width
        factor = 1.0
        if modifiers & key.MOD_SHIFT:
            factor *= 0.1
        if modifiers & (key.MOD_COMMAND | key.MOD_CTRL):
            factor *= 10.0
        normalized = self._drag_start_normalized + delta * factor
        normalized = max(0.0, min(1.0, normalized))
        value = lo + normalized * (hi - lo)
        if self.descriptor.value_type == "int":
            value = int(round(value))
        self.store.set_override(self.descriptor.id, value, source="gui")
        self._drag_start_normalized = normalized
        self._drag_origin_x = x

    def reset(self) -> None:
        self.store.set_override(
            self.descriptor.id,
            self.descriptor.default_value,
            source="gui",
        )

    def _ensure_graphics(self, batch: Batch) -> None:
        if self._batch is not batch:
            self._batch = batch
            self._label = None
            self._value_label = None
            self._track = None
            self._fill = None
            self._dirty_bounds = True
        if self._label is None:
            self._label = pyglet.text.Label(
                text=self.descriptor.label,
                x=self.x,
                y=self.y + self.height - 4,
                anchor_x="left",
                anchor_y="top",
                font_size=self.layout.font_size,
                color=(240, 240, 240, 255),
                batch=batch,
                group=self._label_group,
            )
        if self._value_label is None:
            self._value_label = pyglet.text.Label(
                text="",
                x=self.x + self.width + 8,
                y=self.y + self.height / 2,
                anchor_x="left",
                anchor_y="center",
                font_size=self.layout.font_size,
                color=(210, 210, 210, 255),
                batch=batch,
                group=self._value_group,
            )
        # NOTE: トラックを無効化し fill のみで表示する
        self._track = None
        if self._fill is None:
            self._fill = pyglet.shapes.Rectangle(
                x=self.x,
                y=self.y,
                width=0,
                height=max(self.height * 0.45, 8.0),
                color=(110, 170, 255),
                batch=batch,
                group=self._fill_group,
            )

    def _update_bounds(self) -> None:
        if not self._dirty_bounds:
            return
        if self._label is not None:
            self._label.x = self.x
            self._label.y = self.y + self.height - 4
        if self._value_label is not None:
            self._value_label.x = self.x + self.width + 8
            self._value_label.y = self.y + self.height / 2
        bar_h = max(self.height * 0.45, 8.0)
        bar_y = self.y + (self.height - bar_h) / 2
        # トラックは描画しない
        if self._fill is not None:
            self._fill.x = self.x
            self._fill.y = bar_y
            self._fill.height = bar_h
        self._dirty_bounds = False

    def _update_value(self) -> None:
        value = self.current_value()
        if isinstance(value, float):
            value_text = f"{value:.{self.layout.value_precision}f}"
        else:
            value_text = str(value)
        if self._value_label is not None:
            self._value_label.text = value_text
        normalized = self._normalized()
        if self._fill is not None:
            fill_width = max(2.0, normalized * self.width)
            self._fill.width = min(self.width, fill_width)

    def draw(self, batch: Batch) -> None:
        self._ensure_graphics(batch)
        self._update_bounds()
        self._update_value()

    def hit_test(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


@dataclass
class ToggleWidget:
    """bool パラメータ用のトグルボタン。"""

    descriptor: ParameterDescriptor
    store: ParameterStore
    layout: ParameterLayoutConfig
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    dragging: bool = False

    def __post_init__(self) -> None:
        self._batch: Batch | None = None
        self._label: pyglet.text.Label | None = None
        self._state_label: pyglet.text.Label | None = None
        self._background: pyglet.shapes.Rectangle | None = None
        self._label_group = Group(order=0)
        self._toggle_group = Group(order=1)
        self._state_group = Group(order=2)

    def set_bounds(self, *, x: float, y: float, width: float, height: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def current_value(self) -> bool:
        value = self.store.current_value(self.descriptor.id)
        if value is None:
            return bool(self.descriptor.default_value)
        return bool(value)

    def begin_drag(self) -> None:
        self.dragging = True
        self._toggle()

    def drag_to(self, _x: float, *, modifiers: int = 0) -> None:  # noqa: ARG002
        return None

    def end_drag(self) -> None:
        self.dragging = False

    def reset(self) -> None:
        self.store.set_override(
            self.descriptor.id,
            float(bool(self.descriptor.default_value)),
            source="gui",
        )

    def draw(self, batch: Batch) -> None:
        self._ensure_graphics(batch)
        self._update_bounds()
        self._update_state()

    def hit_test(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def _ensure_graphics(self, batch: Batch) -> None:
        if self._batch is not batch:
            self._batch = batch
            self._label = None
            self._state_label = None
            self._background = None
        if self._label is None:
            self._label = pyglet.text.Label(
                text=self.descriptor.label,
                x=self.x,
                y=self.y + self.height - 4,
                anchor_x="left",
                anchor_y="top",
                font_size=self.layout.font_size,
                color=(240, 240, 240, 255),
                batch=batch,
                group=self._label_group,
            )
        if self._background is None:
            self._background = pyglet.shapes.Rectangle(
                x=self.x,
                y=self.y,
                width=0,
                height=0,
                color=(70, 90, 110),
                batch=batch,
                group=self._toggle_group,
            )
        if self._state_label is None:
            self._state_label = pyglet.text.Label(
                text="",
                x=self.x,
                y=self.y,
                anchor_x="center",
                anchor_y="center",
                font_size=self.layout.font_size,
                color=(240, 240, 240, 255),
                batch=batch,
                group=self._state_group,
            )

    def _update_bounds(self) -> None:
        toggle_width = max(50.0, min(120.0, self.width * 0.45))
        toggle_height = max(self.height * 0.6, 16.0)
        toggle_x = self.x + self.width - toggle_width
        toggle_y = self.y + (self.height - toggle_height) / 2
        if self._label is not None:
            self._label.x = self.x
            self._label.y = self.y + self.height - 4
        if self._background is not None:
            self._background.x = toggle_x
            self._background.y = toggle_y
            self._background.width = toggle_width
            self._background.height = toggle_height
        if self._state_label is not None:
            self._state_label.x = toggle_x + toggle_width / 2
            self._state_label.y = toggle_y + toggle_height / 2

    def _update_state(self) -> None:
        active = self.current_value()
        if self._background is not None:
            self._background.color = (90, 150, 240) if active else (70, 90, 110)
        if self._state_label is not None:
            self._state_label.text = "ON" if active else "OFF"

    def _toggle(self) -> None:
        new_value = not self.current_value()
        self.store.set_override(self.descriptor.id, float(new_value), source="gui")


class ParameterPanel:
    """複数スライダーのレイアウトとイベント処理を担う。"""

    def __init__(self, store: ParameterStore, layout: ParameterLayoutConfig) -> None:
        self._store = store
        self._layout = layout
        self._widgets: list[ParameterWidget] = []
        self._visible_widgets: list[ParameterWidget] = []
        self._scroll_offset = 0.0
        self._viewport_height = 0.0
        self._dirty = True
        self._batch: Batch | None = None

    # --- データ更新 ---
    def update_descriptors(self, descriptors: Sequence[ParameterDescriptor]) -> None:
        remaining = {widget.descriptor.id: widget for widget in self._widgets}
        widgets: list[ParameterWidget] = []
        for desc in descriptors:
            if not desc.supported:
                continue
            widget = remaining.pop(desc.id, None)
            if widget is None or not self._is_widget_compatible(widget, desc):
                widget = self._create_widget(desc)
            widgets.append(widget)
        self._widgets = widgets
        self._dirty = True

    # --- レイアウト ---
    def layout(self, width: float, height: float) -> None:
        if not self._dirty and self._visible_widgets:
            return
        self._batch = Batch()
        self._viewport_height = height
        max_offset = self._max_scroll_offset(height)
        self._scroll_offset = min(self._scroll_offset, max_offset)
        row_height = self._layout.row_height
        y = height - self._layout.padding - row_height
        start_index = int(max(0, self._scroll_offset // row_height))
        offset_px = self._scroll_offset % row_height
        visible: list[ParameterWidget] = []
        for widget in self._widgets[start_index:]:
            if y + row_height < 0:
                break
            widget.set_bounds(
                x=self._layout.padding,
                y=y - offset_px,
                width=max(10.0, width - 3 * self._layout.padding - 80.0),
                height=row_height,
            )
            visible.append(widget)
            y -= row_height
            if y < -row_height:
                break
        self._visible_widgets = visible
        self._dirty = False

    # --- スクロール ---
    def scroll(self, amount: float) -> None:
        max_offset = self._max_scroll_offset(self._viewport_height)
        new_offset = self._scroll_offset + amount * _SCROLL_STEP
        self._scroll_offset = min(max(0.0, new_offset), max_offset)
        self._dirty = True

    # --- 描画 ---
    def draw(self) -> None:
        if self._batch is None:
            self._batch = Batch()
        for widget in self._visible_widgets:
            widget.draw(self._batch)
        self._batch.draw()

    # --- イベント ---
    def hit_test(self, x: float, y: float) -> ParameterWidget | None:
        for widget in self._visible_widgets:
            if widget.hit_test(x, y):
                return widget
        return None

    @property
    def widgets(self) -> Iterable[ParameterWidget]:
        return self._widgets

    def _create_widget(self, descriptor: ParameterDescriptor) -> ParameterWidget:
        if descriptor.value_type == "bool":
            return ToggleWidget(descriptor, self._store, self._layout)
        return SliderWidget(descriptor, self._store, self._layout)

    def _is_widget_compatible(
        self, widget: ParameterWidget, descriptor: ParameterDescriptor
    ) -> bool:
        if descriptor.value_type == "bool":
            return isinstance(widget, ToggleWidget)
        return isinstance(widget, SliderWidget)

    def _max_scroll_offset(self, viewport_height: float) -> float:
        if viewport_height <= 0:
            return 0.0
        content_height = len(self._widgets) * self._layout.row_height
        visible_height = max(0.0, viewport_height - 2 * self._layout.padding)
        if visible_height <= 0:
            return float(content_height)
        return max(0.0, content_height - visible_height)
