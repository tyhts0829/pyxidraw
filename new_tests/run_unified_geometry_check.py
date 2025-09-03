import traceback
import logging
from common.logging import setup_default_logging

from test_unified_geometry import (
    test_from_lines_and_empty,
    test_translate_is_pure,
    test_scale_uniform_and_center,
    test_rotate_z_90deg,
    test_concat_lines,
    test_as_arrays_copy_behavior,
    test_effect_noise_integration_path,
)


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


def main():
    setup_default_logging()
    tests = [
        ("from_lines_and_empty", test_from_lines_and_empty),
        ("translate_is_pure", test_translate_is_pure),
        ("scale_uniform_and_center", test_scale_uniform_and_center),
        ("rotate_z_90deg", test_rotate_z_90deg),
        ("concat_lines", test_concat_lines),
        ("as_arrays_copy_behavior", test_as_arrays_copy_behavior),
        ("effect_noise_integration_path", test_effect_noise_integration_path),
    ]
    ok = True
    for name, fn in tests:
        ok = run(name, fn) and ok
    logger.info("RESULT: %s", "ALL PASSED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
