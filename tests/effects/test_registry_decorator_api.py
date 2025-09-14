from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

from effects import registry as _effects_registry
from effects.registry import clear_registry, effect, get_effect, get_registry, is_effect_registered
from engine.core.geometry import Geometry


@pytest.fixture(scope="module", autouse=True)
def _restore_effects_registry_after_module() -> None:
    """本モジュールのテスト終了後に effects レジストリを復元する。

    - 本モジュール内で `clear_registry()` を呼ぶテストがあるため、他モジュールへの影響を避ける。
    - 方針: `effects.registry` を reload してレジストリを初期化し、`effects/` 配下の
      各サブモジュールを reload/import して `@effect` による登録を再実行する。
    """

    # 実行（yield 後に復元）
    yield

    import effects  # 遅延 import（テスト中に import されている前提）

    # 1) レジストリ本体を再初期化
    importlib.reload(_effects_registry)

    # 2) パッケージ配下のサブモジュールを再ロードして登録を復元
    for mod in pkgutil.iter_modules(effects.__path__):
        fullname = f"{effects.__name__}.{mod.name}"
        if fullname.endswith(".registry"):
            # registry は上で reload 済み
            continue
        if fullname in sys.modules:
            importlib.reload(sys.modules[fullname])
        else:
            importlib.import_module(fullname)

    # 3) パッケージの集約 import を再評価
    importlib.reload(effects)


def identity(g: Geometry) -> Geometry:  # helper for type hints
    return g


def test_effect_decorator_supports_name_keyword() -> None:
    @effect(name="custom_effect")
    def my_fx(g: Geometry) -> Geometry:
        return identity(g)

    assert is_effect_registered("custom_effect")
    fn = get_effect("custom_effect")
    assert callable(fn)
    clear_registry()
    assert not is_effect_registered("custom_effect")


def test_effect_decorator_supports_positional_name() -> None:
    @effect("posfx")
    def my_fx2(g: Geometry) -> Geometry:
        return identity(g)

    assert is_effect_registered("posfx")
    fn = get_effect("posfx")
    assert callable(fn)
    clear_registry()


def test_effect_decorator_rejects_non_function_with_message() -> None:
    class NotFunc:  # noqa: N801 - dummy
        pass

    deco = effect(name="bad")
    with pytest.raises(TypeError) as ei:
        deco(NotFunc)
    assert "got" in str(ei.value)


def test_get_registry_returns_copy() -> None:
    snap = get_registry()
    assert isinstance(snap, dict)
    snap["bogus"] = object()
    assert not is_effect_registered("bogus")
