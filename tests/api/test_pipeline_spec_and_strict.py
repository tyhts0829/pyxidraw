from __future__ import annotations


from api import E


def test_builder_allows_unknown_param_at_build_time() -> None:
    builder = E.pipeline
    # ビルド時は厳格検証を行わない（未知キーは許容）
    builder = builder.rotate(bad_param=123)  # type: ignore[call-arg]
    builder.build()  # 例外にならない


def test_builder_build_no_raise_with_valid_params() -> None:
    ok = E.pipeline.rotate(angles_rad=(0.0, 0.0, 0.0))
    ok.build()  # 例外なし
