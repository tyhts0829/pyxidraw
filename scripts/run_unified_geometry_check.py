#!/usr/bin/env python3
"""
Minimal geometry sanity runner (moved from new_tests/).

Runs a subset of unified-geometry checks without pytest, useful in constrained
environments. It imports from the canonical tests under tests/ (v3 naming).
"""

import logging
import traceback

from common.logging import setup_default_logging

# Import canonical tests from tests/ (v3 suite)
from tests.test_unified_geometry_v3 import (  # type: ignore
    test_from_lines_and_empty,
    test_translate_is_pure,
    test_scale_uniform_and_center,
    test_rotate_z_90deg,
    test_concat_lines,
    test_as_arrays_copy_behavior,
)
from tests.test_noise import test_noise_stability_small_input  # type: ignore


logger = logging.getLogger(__name__)


def run(name, fn):
    try:
        fn()
        logger.info("[PASS] %s", name)
        return True
    except Exception:
        logger.error("[FAIL] %s", name)
        traceback.print_exc()
        return False


def main() -> int:
    setup_default_logging()
    tests = [
        ("from_lines_and_empty", test_from_lines_and_empty),
        ("translate_is_pure", test_translate_is_pure),
        ("scale_uniform_and_center", test_scale_uniform_and_center),
        ("rotate_z_90deg", test_rotate_z_90deg),
        ("concat_lines", test_concat_lines),
        ("as_arrays_copy_behavior", test_as_arrays_copy_behavior),
        ("noise_stability_small_input", test_noise_stability_small_input),
    ]
    ok = True
    for name, fn in tests:
        ok = run(name, fn) and ok
    logger.info("RESULT: %s", "ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
