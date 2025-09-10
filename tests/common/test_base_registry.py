from __future__ import annotations

import pytest

from common.base_registry import BaseRegistry


def test_register_and_get_with_normalization() -> None:
    reg = BaseRegistry()

    @reg.register(None)
    class MyThing:  # noqa: N801 (テスト用)
        pass

    assert reg.is_registered("my_thing")
    assert reg.get("MyThing") is MyThing
    assert "my_thing" in reg.list_all()


def test_duplicate_and_unregister() -> None:
    reg = BaseRegistry()

    @reg.register(None)
    def sample():  # noqa: ANN001 - テスト用
        return 1

    with pytest.raises(ValueError):
        reg.register("sample")(lambda: 2)

    reg.unregister("Sample")
    assert not reg.is_registered("sample")


def test_unregister_noop_and_list_all_contains_registered() -> None:
    reg = BaseRegistry()

    @reg.register(None)
    def a():  # noqa: ANN001 - テスト用
        return 1

    reg.unregister("nonexistent")  # 例外にならない
    names = reg.list_all()
    assert any(n == "a" for n in names)


def test_key_normalization_hyphen_to_snake() -> None:
    reg = BaseRegistry()

    @reg.register("My-Effect")
    def fn():  # noqa: ANN001 - テスト用
        return 0

    # ハイフン→アンダースコア + キャメル→スネークの合成で二重 '_' になる実装（仕様通り）
    assert reg.is_registered("My-Effect")
    assert reg.get("my__effect") is fn
