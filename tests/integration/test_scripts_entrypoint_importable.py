from __future__ import annotations

import importlib

import pytest


@pytest.mark.integration
def test_scripts_gen_g_stubs_entrypoint_importable():
    """scripts.gen_g_stubs must expose generate_stubs_str() and main().

    This guards against refactors of the scripts/ package structure; we avoid invoking
    the real main() to prevent file writes, but ensure the entrypoints are present.
    """
    mod = importlib.import_module("scripts.gen_g_stubs")
    assert hasattr(mod, "generate_stubs_str") and callable(mod.generate_stubs_str)
    assert hasattr(mod, "main") and callable(mod.main)
