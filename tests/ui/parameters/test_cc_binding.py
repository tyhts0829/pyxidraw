import pytest

pytest.importorskip("dearpygui.dearpygui")

import dearpygui.dearpygui as dpg  # type: ignore

from engine.ui.parameters.dpg_window_content import ParameterWindowContentBuilder
from engine.ui.parameters.dpg_window_theme import ParameterWindowThemeManager
from engine.ui.parameters.state import ParameterLayoutConfig, ParameterStore


def _make_builder() -> ParameterWindowContentBuilder:
    store = ParameterStore()
    layout = ParameterLayoutConfig()
    theme_mgr = ParameterWindowThemeManager(layout=layout, theme_cfg=None)
    return ParameterWindowContentBuilder(store=store, layout=layout, theme_mgr=theme_mgr)


def test_on_cc_binding_change_parses_and_clamps():
    builder = _make_builder()
    builder._on_cc_binding_change(sender=0, app_data="140.9", user_data="param.x")
    assert builder._store.cc_binding("param.x") == 127


def test_on_cc_binding_change_clears_on_invalid_input(monkeypatch):
    builder = _make_builder()
    builder._store.bind_cc("param.y", 10)

    recorded: list[tuple[str, str]] = []

    def _set_value(tag, value):  # type: ignore[no-untyped-def]
        recorded.append((str(tag), str(value)))

    monkeypatch.setattr(dpg, "set_value", _set_value)

    builder._on_cc_binding_change(sender=0, app_data="not-a-number", user_data="param.y")

    assert builder._store.cc_binding("param.y") is None
    assert ("param.y::cc", "") in recorded
