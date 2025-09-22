"""
どこで: `engine.ui.parameters` の GUI 表示レイヤ。
何を: ParameterStore の Descriptor をスライダー/トグルへ投影し、描画/入力/レイアウト/スクロールを扱う。
なぜ: 軽量なウィジェットでパラメータ編集を可能にし、ヘッドレスでも動作可能な設計を保つため。

補足:
- スライダーのバー表示は表示上の比率にクランプするが、内部の実値はクランプしない。
- 表示比率は実レンジの線形変換 `(value - min)/(max - min)` を用いる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import pyglet
import pyglet.shapes
from pyglet.graphics import Batch, Group
from pyglet.window import key

from .state import ParameterDescriptor, ParameterLayoutConfig, ParameterStore, RangeHint

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

    def _hint(self) -> RangeHint | None:
        return self.descriptor.range_hint

    def _current_actual(self) -> Any:
        value = self.store.current_value(self.descriptor.id)
        if value is None:
            value = self.descriptor.default_value
        if self.descriptor.value_type == "int":
            return int(round(float(value)))
        if self.descriptor.value_type == "bool":
            return bool(value)
        return float(value) if isinstance(value, (int, float)) else value

    def _normalized(self) -> float:
        # 表示用の 0..1 比率
        hint = self._hint()
        value = float(self._current_actual())
        if hint is None:
            return 0.0
        lo = float(hint.min_value)
        hi = float(hint.max_value)
        span = max(hi - lo, 1e-9)
        ratio = (value - lo) / span
        # 表示上のみクランプ
        return max(0.0, min(1.0, ratio))

    def begin_drag(self) -> None:
        self.dragging = True
        self._drag_start_normalized = self._normalized()
        self._drag_origin_x = self.x + self._drag_start_normalized * max(self.width, 1.0)

    def end_drag(self) -> None:
        self.dragging = False

    def drag_to(self, x: float, *, modifiers: int = 0) -> None:
        width = max(self.width, 1.0)
        delta = (x - self._drag_origin_x) / width
        factor = 1.0
        if modifiers & key.MOD_SHIFT:
            factor *= 0.1
        if modifiers & (key.MOD_COMMAND | key.MOD_CTRL):
            factor *= 10.0
        new_ratio = self._drag_start_normalized + delta * factor
        hint = self._hint()
        if hint is None:
            return
        # 表示上のみクランプ
        new_ratio = max(0.0, min(1.0, float(new_ratio)))
        lo = float(hint.min_value)
        hi = float(hint.max_value)
        actual_value: Any = lo + (hi - lo) * new_ratio
        # step/丸め
        if hint.step is not None:
            step = float(hint.step)
            if step > 0:
                actual_value = lo + round((actual_value - lo) / step) * step
        if self.descriptor.value_type == "int":
            actual_value = int(round(float(actual_value)))
        self.store.set_override(self.descriptor.id, actual_value, source="gui")
        self._drag_start_normalized = float(new_ratio)
        self._drag_origin_x = x

    def reset(self) -> None:
        self.store.set_override(self.descriptor.id, self.descriptor.default_value, source="gui")

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
        # ラベルは行の上側に表示（重ねない）
        label_h = float(self.layout.value_precision)  # temp init
        if self._label is not None:
            self._label.x = self.x
            self._label.y = self.y + self.height - 4
            try:
                label_h = float(max(self._label.content_height, self.layout.font_size))
            except Exception:
                label_h = float(self.layout.font_size)
        else:
            label_h = float(self.layout.font_size)

        # ラベル分の高さを差し引いてバーを配置
        label_gap = 6.0
        usable_h = max(0.0, self.height - label_h - label_gap)
        bar_h = min(max(self.height * 0.45, 8.0), usable_h) if usable_h > 0 else 0.0
        bar_y = self.y + max(0.0, (usable_h - bar_h) / 2)

        if self._value_label is not None:
            self._value_label.x = self.x + self.width + 8
            self._value_label.y = bar_y + bar_h / 2 if bar_h > 0 else self.y
        # トラックは描画しない（現状）
        if self._fill is not None:
            self._fill.x = self.x
            self._fill.y = bar_y
            self._fill.height = bar_h
        self._dirty_bounds = False

    def _update_value(self) -> None:
        value = self._current_actual()
        if isinstance(value, float):
            value_text = f"{value:.{self.layout.value_precision}f}"
        elif isinstance(value, bool):
            value_text = "ON" if value else "OFF"
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
            self.descriptor.id, bool(self.descriptor.default_value), source="gui"
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
        # ラベルは上、トグルは全幅でその下に配置
        label_h = float(self.layout.font_size)
        if self._label is not None:
            self._label.x = self.x
            self._label.y = self.y + self.height - 4
            try:
                label_h = float(max(self._label.content_height, self.layout.font_size))
            except Exception:
                label_h = float(self.layout.font_size)
        label_gap = 6.0
        usable_h = max(0.0, self.height - label_h - label_gap)
        toggle_height = max(16.0, usable_h * 0.8)
        toggle_x = self.x
        toggle_y = self.y + max(0.0, (usable_h - toggle_height) / 2)
        toggle_width = self.width
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
        self.store.set_override(self.descriptor.id, bool(new_value), source="gui")


@dataclass
class EnumWidget:
    """enum パラメータ用のセグメント選択ウィジェット。

    - `choices` の配列を右側に等幅で描画し、クリック位置で選択する。
    - 単一クリックで `drag_to(x)` が呼ばれるため、そこでインデックスを決定。
    """

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
        self._segment_rects: list[pyglet.shapes.Rectangle] = []
        self._segment_labels: list[pyglet.text.Label] = []
        self._label_group = Group(order=0)
        self._seg_bg_group = Group(order=1)
        self._seg_text_group = Group(order=2)

    def set_bounds(self, *, x: float, y: float, width: float, height: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def _current_value(self) -> str | None:
        value = self.store.current_value(self.descriptor.id)
        if value is None:
            dv = self.descriptor.default_value
            return str(dv) if dv is not None else None
        return str(value)

    def _selected_index(self) -> int:
        choices = self.descriptor.choices or []
        current = self._current_value()
        if current is None:
            return -1
        try:
            return choices.index(current)
        except ValueError:
            return -1

    def begin_drag(self) -> None:
        self.dragging = True

    def drag_to(self, x: float, *, modifiers: int = 0) -> None:  # noqa: ARG002
        choices = self.descriptor.choices or []
        if not choices:
            return
        seg_area_w = max(self.width * 0.6, 1.0)
        seg_area_x = self.x + self.width - seg_area_w
        seg_w = max(seg_area_w / len(choices), 1.0)
        idx = int((x - seg_area_x) // seg_w)
        if idx < 0 or idx >= len(choices):
            return
        self.store.set_override(self.descriptor.id, str(choices[idx]), source="gui")

    def end_drag(self) -> None:
        self.dragging = False

    def reset(self) -> None:
        dv = self.descriptor.default_value
        if dv is None and self.descriptor.choices:
            dv = self.descriptor.choices[0]
        self.store.set_override(self.descriptor.id, str(dv), source="gui")

    def draw(self, batch: Batch) -> None:
        self._ensure_graphics(batch)
        self._update_layout()
        self._update_state()

    def hit_test(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def _ensure_graphics(self, batch: Batch) -> None:
        if self._batch is not batch:
            self._batch = batch
            self._label = None
            self._segment_rects = []
            self._segment_labels = []
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
        # セグメントを choices 数に合わせて再構築
        need_n = len(self.descriptor.choices or [])
        if len(self._segment_rects) != need_n:
            # 既存の参照は pyglet の Batch が破棄を管理するため上書きで良い
            self._segment_rects = []
            self._segment_labels = []
            for _ in range(need_n):
                self._segment_rects.append(
                    pyglet.shapes.Rectangle(
                        x=0,
                        y=0,
                        width=0,
                        height=0,
                        color=(70, 90, 110),
                        batch=batch,
                        group=self._seg_bg_group,
                    )
                )
                self._segment_labels.append(
                    pyglet.text.Label(
                        text="",
                        x=0,
                        y=0,
                        anchor_x="center",
                        anchor_y="center",
                        font_size=self.layout.font_size,
                        color=(220, 220, 220, 255),
                        batch=batch,
                        group=self._seg_text_group,
                    )
                )

    def _update_layout(self) -> None:
        # パラメータラベルはバーの少し上に表示（重ねない）
        label_height = float(self.layout.font_size)
        if self._label is not None:
            self._label.x = self.x
            self._label.y = self.y + self.height - 4
            try:
                label_height = float(max(self._label.content_height, self.layout.font_size))
            except Exception:
                label_height = float(self.layout.font_size)

        choices = self.descriptor.choices or []
        # EnumWidget は行幅いっぱいを使用
        seg_area_w = max(self.width, 1.0)
        seg_area_x = self.x
        seg_h = max(self.height * 0.5, 12.0)
        # ラベル分の高さを上側に確保してから、残り領域でバーを配置
        label_gap = 4.0
        usable_h = max(0.0, self.height - label_height - label_gap)
        seg_y = self.y + max(0.0, (usable_h - seg_h) / 2)
        n = len(choices)
        if n == 0:
            return

        # 1) 各ラベルの実テキスト幅を取得し、左右パディングを加えた希望幅を算出
        gap = 4.0
        pad = 12.0  # 左右合計のパディング
        desired: list[float] = []
        for i in range(n):
            lbl = self._segment_labels[i]
            text = str(choices[i])
            if lbl.text != text:
                lbl.text = text
            try:
                text_w = float(lbl.content_width)
            except Exception:
                # フォールバック（大雑把な幅推定）
                text_w = max(1.0, len(text) * self.layout.font_size * 0.6)
            desired.append(max(1.0, text_w + pad))

        total_needed = sum(desired) + gap * (n - 1)

        # 2) 可用幅に収まる場合は希望幅で配置。収まらない場合は等分幅を使用（フォールバック）。
        if total_needed <= seg_area_w:
            x_cursor = seg_area_x
            for i in range(n):
                rect = self._segment_rects[i]
                w = desired[i]
                rect.x = x_cursor
                rect.y = seg_y
                rect.width = w
                rect.height = seg_h
                lbl = self._segment_labels[i]
                lbl.x = rect.x + rect.width / 2
                lbl.y = rect.y + rect.height / 2
                x_cursor += w + gap
        else:
            # 収まらない場合は等分（後続タスクでドロップダウンに切替予定）
            seg_w = max((seg_area_w - gap * (n - 1)) / n, 1.0)
            x_cursor = seg_area_x
            for i in range(n):
                rect = self._segment_rects[i]
                rect.x = x_cursor
                rect.y = seg_y
                rect.width = seg_w
                rect.height = seg_h
                lbl = self._segment_labels[i]
                lbl.x = rect.x + rect.width / 2
                lbl.y = rect.y + rect.height / 2
                x_cursor += seg_w + gap

    def _update_state(self) -> None:
        sel = self._selected_index()
        for i, rect in enumerate(self._segment_rects):
            rect.color = (90, 150, 240) if i == sel else (70, 90, 110)


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
            # Enum は行幅いっぱいを使う。その他は既存の幅配分。
            if isinstance(widget, (EnumWidget, ToggleWidget)):
                w = max(10.0, width - 2 * self._layout.padding)
            else:
                w = max(10.0, width - 3 * self._layout.padding - 80.0)
            widget.set_bounds(
                x=self._layout.padding,
                y=y - offset_px,
                width=w,
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
        if descriptor.value_type == "enum":
            return EnumWidget(descriptor, self._store, self._layout)
        return SliderWidget(descriptor, self._store, self._layout)

    def _is_widget_compatible(
        self, widget: ParameterWidget, descriptor: ParameterDescriptor
    ) -> bool:
        if descriptor.value_type == "bool":
            return isinstance(widget, ToggleWidget)
        if descriptor.value_type == "enum":
            return isinstance(widget, EnumWidget)
        return isinstance(widget, SliderWidget)

    def _max_scroll_offset(self, viewport_height: float) -> float:
        if viewport_height <= 0:
            return 0.0
        content_height = len(self._widgets) * self._layout.row_height
        visible_height = max(0.0, viewport_height - 2 * self._layout.padding)
        if visible_height <= 0:
            return float(content_height)
        return max(0.0, content_height - visible_height)
