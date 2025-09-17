"""ParameterWindow を制御する Facade。"""

from __future__ import annotations

from typing import Mapping

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
        self._window: ParameterWindow | None = None
        self._visible: bool = True

    def start(self) -> None:
        if self._window is None and self._visible:
            self._window = ParameterWindow(store=self._store, layout=self._layout)  # type: ignore[abstract]

    def tick(self, _dt: float) -> None:
        # 現状は pyglet 側の schedule に委ねるため処理なし
        return None

    def apply_overrides(self, cc_snapshot: Mapping[int, float]) -> Mapping[int, float]:
        # 現状は GUI と CC の直接連携は未実装のため情報をそのまま返す。
        return dict(cc_snapshot)

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
    def window(self) -> ParameterWindow | None:
        return self._window
