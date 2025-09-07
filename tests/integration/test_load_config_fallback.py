from __future__ import annotations

import pytest

from util.utils import load_config


@pytest.mark.integration
# What this tests (TEST_HARDENING_PLAN.md §Runner/設定)
# - load_config merges configs/default.yaml (base) with root config.yaml (override).
def test_load_config_merges_default_and_root():
    cfg = load_config()
    # configs/default.yaml provides `test_marker: true` which should appear
    assert cfg.get("test_marker") is True
