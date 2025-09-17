"""Parameter GUI 管理ユーティリティ（内部使用）。"""

from __future__ import annotations

from typing import Callable, Mapping

from engine.core.geometry import Geometry

from .controller import ParameterWindowController
from .runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from .state import ParameterLayoutConfig, ParameterStore


class ParameterManager:
    """`user_draw` をラップして ParameterRuntime を介在させる内部ヘルパー。"""

    def __init__(
        self,
        user_draw: Callable[[float, Mapping[int, float]], Geometry],
        *,
        layout: ParameterLayoutConfig | None = None,
        lazy_trace: bool = True,
    ) -> None:
        self._user_draw = user_draw
        self.store = ParameterStore()
        self.runtime = ParameterRuntime(self.store, layout=layout)
        self.runtime.set_lazy(lazy_trace)
        self.controller = ParameterWindowController(self.store, layout=layout)
        self._initialized = False

    def initialize(self, cc_snapshot: Mapping[int, float]) -> None:
        if self._initialized:
            return
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            self._user_draw(0.0, cc_snapshot)
        finally:
            deactivate_runtime()
        descriptors = self.store.descriptors()
        if descriptors:
            self.controller.start()
        else:
            self.controller.set_visibility(False)
        self._initialized = True

    def draw(self, t: float, cc_values: Mapping[int, float]) -> Geometry:
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            merged_cc = self.controller.apply_overrides(cc_values)
            return self._user_draw(t, merged_cc)
        finally:
            deactivate_runtime()

    def shutdown(self) -> None:
        self.controller.shutdown()
