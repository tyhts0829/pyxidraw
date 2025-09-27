"""
どこで: `engine.ui.parameters` の統合ヘルパ層。
何を: `user_draw` をラップし、ParameterRuntime の有効化/初回トレース/GUI ウィンドウ起動・寿命管理を担う。
なぜ: 既存の描画関数に最小介入でパラメータランタイム/GUI を組み込むため。
"""

from __future__ import annotations

from typing import Callable

from engine.core.geometry import Geometry

from .controller import ParameterWindowController
from .runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from .state import ParameterLayoutConfig, ParameterStore


class ParameterManager:
    """`user_draw` をラップして ParameterRuntime を介在させる内部ヘルパー。"""

    def __init__(
        self,
        user_draw: Callable[[float], Geometry],
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

    def initialize(self) -> None:
        if self._initialized:
            return
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            # 初回は t=0 のみ登録（cc は api.cc 側に閉じる）
            self.runtime.set_inputs(0.0)
            self._user_draw(0.0)
        finally:
            deactivate_runtime()
        descriptors = self.store.descriptors()
        if descriptors:
            self.controller.start()
        else:
            self.controller.set_visibility(False)
        self._initialized = True

    def draw(self, t: float) -> Geometry:
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            # 現在の CC は api.cc に閉じる（GUI は Store 経由）
            self.runtime.set_inputs(t)
            return self._user_draw(t)
        finally:
            deactivate_runtime()

    def shutdown(self) -> None:
        self.controller.shutdown()
