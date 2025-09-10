from __future__ import annotations

import pytest

from api.shape_registry import get_shape_generator, unregister_shape


def test_unregister_shape_ignores_missing() -> None:
    # 存在しない名前でも例外を出さない
    unregister_shape("__does_not_exist__")


def test_get_shape_generator_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_shape_generator("__does_not_exist__")
