"""
どこで: `engine.ui.parameters` の状態管理層。
何を: ParameterDescriptor/RangeHint のメタと、ParameterStore による値（original/override/midi）を集中管理。
    購読通知と簡易レンジ推定（ParameterLayoutConfig）も提供。
なぜ: UI/ランタイムが共有する単一の真実源（Single Source of Truth）として、一貫した状態管理を担うため。
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Iterable, Literal

ValueType = Literal["float", "int", "bool", "enum", "vector"]
SourceType = Literal["shape", "effect"]
OverrideSource = Literal["gui", "midi"]


@dataclass(frozen=True)
class RangeHint:
    """UI 表示用の範囲ヒント。"""

    min_value: float | int
    max_value: float | int
    step: float | int | None = None
    scale: str | None = None  # linear / log 等（現状は linear 固定）
    mapped_min: float | int | None = None
    mapped_max: float | int | None = None
    mapped_step: float | int | None = None


@dataclass(frozen=True)
class ParameterDescriptor:
    """GUI に表示するパラメータのメタ情報。"""

    id: str
    label: str
    source: SourceType
    category: str
    value_type: ValueType
    default_value: Any
    range_hint: RangeHint | None = None
    help_text: str | None = None
    vector_group: str | None = None
    supported: bool = True


@dataclass
class ParameterValue:
    """最新の値・上書き状態を保持する。"""

    original: Any
    override: Any | None = None
    midi_override: Any | None = None
    timestamp: float = field(default_factory=time.time)

    def resolve(self) -> Any:
        if self.override is not None:
            return self.override
        if self.midi_override is not None:
            return self.midi_override
        return self.original


@dataclass(frozen=True)
class OverrideResult:
    """override 操作の結果を GUI 側へ返す。"""

    value: Any
    clamped: bool
    source: OverrideSource


Subscriber = Callable[[Iterable[str]], None]


class ParameterStore:
    """パラメータメタデータと値を集中管理する。"""

    def __init__(self) -> None:
        self._descriptors: dict[str, ParameterDescriptor] = {}
        self._values: dict[str, ParameterValue] = {}
        self._listeners: list[Subscriber] = []
        self._lock = RLock()
        self._last_notification: float = 0.0

    # --- 登録 / 問合せ ---
    def register(self, descriptor: ParameterDescriptor, value: Any) -> None:
        """Descriptor を登録し、初期値を保存する。"""
        changed: set[str] = set()
        with self._lock:
            if descriptor.id not in self._descriptors:
                self._descriptors[descriptor.id] = descriptor
                changed.add(descriptor.id)
            if descriptor.id not in self._values:
                self._values[descriptor.id] = ParameterValue(original=value)
                changed.add(descriptor.id)
            else:
                current = self._values[descriptor.id]
                if current.original != value:
                    current.original = value
                    current.timestamp = time.time()
                    changed.add(descriptor.id)
        if changed:
            self._notify(changed)

    def descriptors(self) -> list[ParameterDescriptor]:
        with self._lock:
            return list(self._descriptors.values())

    def get_descriptor(self, param_id: str) -> ParameterDescriptor:
        with self._lock:
            return self._descriptors[param_id]

    # --- 値操作 ---
    def resolve(self, param_id: str, original: Any) -> Any:
        """元値を更新しつつ override を適用した値を返す。"""
        with self._lock:
            if param_id not in self._values:
                self._values[param_id] = ParameterValue(original=original)
                value = self._values[param_id]
            else:
                value = self._values[param_id]
                if value.original != original:
                    value.original = original
                    value.timestamp = time.time()
        return value.resolve()

    def current_value(self, param_id: str) -> Any:
        with self._lock:
            entry = self._values.get(param_id)
            if entry is None:
                return None
            return entry.resolve()

    def original_value(self, param_id: str) -> Any:
        with self._lock:
            entry = self._values.get(param_id)
            if entry is None:
                return None
            return entry.original

    def set_override(
        self,
        param_id: str,
        value: Any,
        *,
        source: OverrideSource = "gui",
    ) -> OverrideResult:
        descriptor = self._descriptors.get(param_id)
        clamped = False
        if descriptor and descriptor.range_hint:
            lo = descriptor.range_hint.min_value
            hi = descriptor.range_hint.max_value
            if (
                isinstance(value, (int, float))
                and isinstance(lo, (int, float))
                and isinstance(hi, (int, float))
            ):
                clamped_value = min(max(value, lo), hi)
                clamped = clamped_value != value
                value = clamped_value
        with self._lock:
            entry = self._values.setdefault(param_id, ParameterValue(original=value))
            if source == "gui":
                entry.override = value
            else:
                entry.midi_override = value
            entry.timestamp = time.time()
        self._notify({param_id})
        return OverrideResult(value=value, clamped=clamped, source=source)

    def clear_override(self, param_id: str, *, source: OverrideSource = "gui") -> None:
        with self._lock:
            entry = self._values.get(param_id)
            if entry is None:
                return
            changed = False
            if source == "gui" and entry.override is not None:
                entry.override = None
                changed = True
            if source == "midi" and entry.midi_override is not None:
                entry.midi_override = None
                changed = True
            if changed:
                entry.timestamp = time.time()
        if changed:
            self._notify({param_id})

    def clear_all_overrides(self) -> None:
        with self._lock:
            changed: set[str] = set()
            for param_id, value in self._values.items():
                if value.override is not None or value.midi_override is not None:
                    value.override = None
                    value.midi_override = None
                    value.timestamp = time.time()
                    changed.add(param_id)
        if changed:
            self._notify(changed)

    # --- リスナー ---
    def subscribe(self, listener: Subscriber) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Subscriber) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _notify(self, param_ids: Iterable[str]) -> None:
        ids = list(param_ids)
        if not ids:
            return
        for listener in list(self._listeners):
            try:
                listener(ids)
            except Exception:
                continue

    # --- デバッグ/ユーティリティ ---
    def dump_state(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for key, desc in self._descriptors.items():
                value = self._values.get(key)
                result[key] = {
                    "descriptor": desc,
                    "value": value,
                }
            return result


@dataclass(frozen=True)
class ParameterLayoutConfig:
    """GUI 表示用レイアウト設定。"""

    row_height: int = 28
    padding: int = 8
    font_size: int = 12
    value_precision: int = 3
    default_range_multiplier: float = 1.0

    def derive_range(self, *, name: str, value_type: ValueType, default_value: Any) -> RangeHint:
        """値ヒューリスティックから範囲を推定する。"""
        multiplier = self.default_range_multiplier
        if value_type == "bool":
            return RangeHint(0, 1, step=1)
        if value_type == "enum":
            return RangeHint(0, 1)
        if value_type in {"float", "vector"}:
            base = float(default_value) if default_value is not None else 0.0
            if base == 0.0:
                lo, hi = 0.0, 1.0
            else:
                delta = max(abs(base), 1.0) * multiplier
                lo, hi = base - delta, base + delta
            if any(token in name for token in ("scale", "subdiv", "freq")):
                lo = max(0.0, lo)
            return RangeHint(lo, hi)
        if value_type == "int":
            base_int = int(default_value or 0)
            delta_int = max(abs(base_int), 1)
            lo = base_int - delta_int
            hi = base_int + delta_int
            if any(token in name for token in ("count", "segments", "subdiv")):
                lo = max(0, lo)
            return RangeHint(lo, hi, step=1)
        return RangeHint(0, 1)


class ParameterRegistry:
    """call ごとの出現回数を追跡するためのユーティリティ。"""

    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)

    def next_index(self, key: str) -> int:
        current = self._counters[key]
        self._counters[key] = current + 1
        return current

    def reset(self) -> None:
        self._counters.clear()
