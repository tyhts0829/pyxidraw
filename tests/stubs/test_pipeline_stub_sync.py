from __future__ import annotations

from tools.gen_g_stubs import generate_stubs_str


def test_pipeline_builder_protocol_contains_registered_effects() -> None:
    s = generate_stubs_str()
    # 代表として rotate/displace/offset あたりのメソッドが含まれること
    assert "class _PipelineBuilder" in s
    for name in ("rotate", "displace", "offset"):
        assert f"def {name}(self, " in s
