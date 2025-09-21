from __future__ import annotations

import pytest

from api import E


def test_strict_unknown_param_raises() -> None:
    builder = E.pipeline  # strict=True が既定
    builder = builder.rotate(bad_param=123)  # type: ignore[call-arg]
    with pytest.raises(TypeError) as ei:
        builder.build()
    assert "unknown params" in str(ei.value)
    assert "Allowed:" in str(ei.value)


def test_builder_strictness_is_enforced_via_signature() -> None:
    # validate_spec は削除されたため、ビルダーの strict 検証を確認する
    ok = E.pipeline.rotate(angles_rad=(0.0, 0.0, 0.0))
    ok.build()  # 例外なし
