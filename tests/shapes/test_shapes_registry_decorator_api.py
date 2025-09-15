from __future__ import annotations

import pytest

from engine.core.geometry import Geometry
from shapes.registry import get_registry, is_shape_registered, shape, unregister


def test_shape_decorator_supports_name_keyword() -> None:
    # 明示名をキーワードで指定して登録
    @shape(name="custom_test_shape")
    def custom_test_shape(**params: object) -> Geometry:
        return Geometry.from_lines([])

    assert is_shape_registered("custom_test_shape")
    unregister("custom_test_shape")
    assert not is_shape_registered("custom_test_shape")


def test_shape_decorator_supports_positional_name() -> None:
    # 明示名を位置引数で指定して登録
    @shape("positional_named_shape")
    def positional_named_shape(**params: object) -> Geometry:
        return Geometry.from_lines([])

    assert is_shape_registered("positional_named_shape")
    unregister("positional_named_shape")
    assert not is_shape_registered("positional_named_shape")


def test_shape_decorator_rejects_non_function_with_message() -> None:
    # 非関数を登録しようとすると TypeError。
    class NotFunc:  # noqa: N801 (テスト用の簡易クラス)
        pass

    deco = shape(name="bad")
    with pytest.raises(TypeError) as ei:
        deco(NotFunc)
    # メッセージが対象オブジェクト情報を含む
    assert "got" in str(ei.value)


def test_get_registry_returns_copy() -> None:
    snap = get_registry()
    assert isinstance(snap, dict)
    # 変更しても内部状態に影響しないこと
    snap["bogus"] = object()
    assert not is_shape_registered("bogus")
