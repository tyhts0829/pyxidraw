import pytest

from api.effects import E
from engine.ui.parameters import (
    ParameterRuntime,
    ParameterStore,
    activate_runtime,
    deactivate_runtime,
)


@pytest.mark.smoke
def test_pipeline_builder_bypass_explicit_no_runtime() -> None:
    # ランタイム無効時でも、明示引数の bypass=True でステップは追加されない
    deactivate_runtime()
    pb = E.pipeline
    b = pb.scale(bypass=True)
    p = b.build()
    assert len(p.steps) == 0


@pytest.mark.smoke
def test_pipeline_builder_bypass_via_gui_runtime() -> None:
    # ランタイム有効時、ParameterStore の override により bypass=True を適用できる
    store = ParameterStore()
    rt = ParameterRuntime(store)
    activate_runtime(rt)
    try:
        rt.begin_frame()
        rt.set_inputs(0.0)
        # 最初のパイプライン UID は p0、最初のステップ index は 0 になる
        # 先に override をセット（Descriptor 未登録でも構わない設計）
        store.set_override("effect@p0.scale#0.bypass", True)
        b = E.pipeline.scale()
        p = b.build()
        assert len(p.steps) == 0
        # Descriptor の登録も行われている（id の存在確認）
        ids = {d.id for d in store.descriptors()}
        assert "effect@p0.scale#0.bypass" in ids
    finally:
        deactivate_runtime()
