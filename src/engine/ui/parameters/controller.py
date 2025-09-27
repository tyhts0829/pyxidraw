"""
どこで: `engine.ui.parameters` 層のウィンドウ制御。
何を: ParameterWindow の生成/可視状態/適用を司る薄いファサード。
なぜ: ウィンドウ寿命管理をロジックから分離し、GUI 有無に依らず安全に扱うため。
"""

from __future__ import annotations

from typing import Any

from .state import ParameterLayoutConfig, ParameterStore
from .window import ParameterWindow


class ParameterWindowController:
    """ParameterWindow のライフサイクルと適用を管理する。"""

    def __init__(
        self,
        store: ParameterStore,
        *,
        layout: ParameterLayoutConfig | None = None,
    ) -> None:
        self._store = store
        self._layout = layout or ParameterLayoutConfig()
        self._window: Any | None = None
        self._visible: bool = True

    def start(self) -> None:
        if self._window is None and self._visible:
            self._window = ParameterWindow(store=self._store, layout=self._layout)  # type: ignore[abstract]

    # tick は不要（DPG はバックグラウンド駆動）

    # cc と GUI のマージは行わない（cc は api 側に限定）。GUI は Store の override で反映される。

    def set_visibility(self, visible: bool) -> None:
        self._visible = visible
        if not visible and self._window is not None:
            self._window.set_visible(False)
        elif visible:
            self.start()
            if self._window is not None:
                self._window.set_visible(True)

    def shutdown(self) -> None:
        if self._window is not None:
            self._window.close()
            self._window = None

    @property
    def window(self) -> Any | None:
        return self._window
