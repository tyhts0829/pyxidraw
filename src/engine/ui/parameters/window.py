"""
どこで: `engine.ui.parameters` のウィンドウ実装（Dear PyGui 版）。
何を: DPG 実装 `ParameterWindow` を公開する薄い再エクスポート。
なぜ: 参照者（Controller など）の import 経路を維持しつつ実装を差し替えるため。
"""

from .dpg_window import ParameterWindow

__all__ = ["ParameterWindow"]
