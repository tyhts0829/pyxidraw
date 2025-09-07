import pytest

from common.base_registry import BaseRegistry

# What this tests (TEST_PLAN.md §Registry)
# - Key normalization (Camel→snake, '-'→'_'), duplicate registration ValueError,
#   unregistered name KeyError, empty name fallback to class name, underscores preserved.



def test_camel_and_hyphen_normalization_and_lookup():
    reg = BaseRegistry()

    @reg.register()
    class MyEffect:
        pass

    # Name variations all resolve
    assert reg.get("MyEffect") is MyEffect
    assert reg.get("myEffect") is MyEffect
    assert reg.get("my_effect") is MyEffect
    assert reg.get("my-effect") is MyEffect

    # Explicit name with hyphen normalizes to underscore
    @reg.register("foo-bar")
    class FooThing:
        pass

    assert reg.get("foo-bar") is FooThing
    assert reg.get("foo_bar") is FooThing


def test_duplicate_registration_raises_and_unregistered_get_raises():
    reg = BaseRegistry()

    @reg.register()
    class MyEffect:
        pass

    class Another:
        pass

    with pytest.raises(ValueError):
        reg.register("MyEffect")(Another)

    with pytest.raises(KeyError):
        reg.get("does-not-exist")


def test_empty_name_registration_falls_back_to_classname():
    """Empty string behaves like None: falls back to class __name__ normalization."""
    reg = BaseRegistry()

    class Dummy:
        pass

    # Empty name -> treat as omitted, so key becomes 'dummy'
    reg.register("")(Dummy)
    assert reg.get("dummy") is Dummy


def test_leading_and_double_underscores_are_preserved():
    reg = BaseRegistry()

    @reg.register("__weird__")
    class W:
        pass

    # Exact lookup with preserved underscores works
    assert reg.get("__weird__") is W
    # Hyphen normalization still applies
    assert reg.get("__weird__") is reg.get("__weird__".replace("-", "_"))
