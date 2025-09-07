from __future__ import annotations

import logging
import time
import warnings

import pytest

from api import E, G
from tests._utils.perf import load_baseline, maybe_update_baseline

logger = logging.getLogger(__name__)


@pytest.mark.perf
# What this tests (TEST_HARDENING_PLAN.md §Perf)
# - Measures a tiny representative pipeline and compares against a JSON baseline.
# - Regressions >30% raise a warning (do not fail); baseline can be updated with env.
def test_perf_smoke_polygon_fill():
    """Measure a small representative pipeline end-to-end.

    - Never fails; only logs and optionally warns on >30% regression.
    - Baseline is persisted under tests/_snapshots/perf when PXD_UPDATE_SNAPSHOTS=1.
    """
    # Representative but light workload (keep this very small to stay stable).
    g = G.polygon(n_sides=12)
    p = (
        E.pipeline.rotate(pivot=(0.0, 0.0, 0.0), angles_rad=(0.0, 0.0, 0.25))
        .fill(mode="lines", density=0.35)
        .build()
    )

    # Warm-up (JIT-like caches, import side-effects, etc.).
    _ = p(g)

    # Time a few runs and take the minimum to reduce noise.
    runs = 3
    times: list[float] = []
    for _i in range(runs):
        t0 = time.perf_counter()
        _ = p(g)
        times.append(time.perf_counter() - t0)

    measured = min(times)

    name = "perf_smoke_polygon_fill_v1"
    baseline = load_baseline(name)
    maybe_update_baseline(name, measured)

    if baseline is not None:
        ratio = measured / baseline if baseline > 0 else float("inf")
        msg = (
            f"perf[{name}] measured={measured*1e3:.2f}ms baseline={baseline*1e3:.2f}ms "
            f"ratio={ratio:.2f}x"
        )
        # Soft regression signal: warn but do not fail.
        if ratio > 1.30:
            warnings.warn("Performance regression >30%: " + msg)
        else:
            logger.info(msg)
    else:
        logger.info(
            "perf[%s] measured=%.2fms (no baseline). Set PXD_UPDATE_SNAPSHOTS=1 to record.",
            name,
            measured * 1e3,
        )
