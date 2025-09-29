import pytest

from engine.ui.parameters.dpg_window import ParameterWindow
from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    RangeHint,
)

try:

    _DPG_AVAILABLE = True
except Exception:  # pragma: no cover - headless/未導入
    _DPG_AVAILABLE = False


@pytest.mark.skipif(_DPG_AVAILABLE, reason="Skip when Dear PyGui is installed (opens window)")
def test_dpg_parameter_window_mounts_and_closes():
    store = ParameterStore()
    layout = ParameterLayoutConfig()
    # 最小の descriptor を登録（float, 0..1）
    desc = ParameterDescriptor(
        id="shape.demo#0.value",
        label="demo · value",
        source="shape",
        category="shape",
        value_type="float",
        default_value=0.5,
        range_hint=RangeHint(0.0, 1.0),
    )
    store.register(desc, 0.5)

    # Dear PyGui 未導入環境ではスタブが使われるため、いずれでも例外なく生成できることを確認
    win = ParameterWindow(store=store, layout=layout)
    # 可視状態の切り替えが例外なく動作すること
    win.set_visible(True)
    win.set_visible(False)
    # 明示的 close が例外なく動作すること
    win.close()
