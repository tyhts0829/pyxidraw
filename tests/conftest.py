from __future__ import annotations

import hashlib
from typing import Callable, Dict, Mapping

import numpy as np
import pytest

from engine.core.geometry import Geometry

# Test scaffolding (fixtures/utilities) for the whole test suite.
# - Based on TEST_PLAN.md: fixtures for reproducibility (rng, digest_hex),
#   geometry seeds (tiny_geom, grid_geom), and environment toggles (env_digest_on/off).
# - Based on TEST_HARDENING_PLAN.md: snapshot helper for e2e/perf and optional
#   dependency guards (require_moderngl/pyglet/numba).


# ---- Optional dependency helpers -------------------------------------------


@pytest.fixture
def require_moderngl():
    import pytest as _pytest

    def _require():
        return _pytest.importorskip("moderngl")

    return _require


@pytest.fixture
def require_pyglet():
    import pytest as _pytest

    def _require():
        return _pytest.importorskip("pyglet")

    return _require


@pytest.fixture
def require_numba():
    import pytest as _pytest

    def _require():
        return _pytest.importorskip("numba")

    return _require


# ---- RNG fixture ------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---- Geometry fixtures ------------------------------------------------------


@pytest.fixture
def tiny_geom() -> Geometry:
    """A very small geometry with mixed 2D/3D inputs for normalization tests."""
    l1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # 2D -> Z=0
    l2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)  # 1D -> reshape(-1,3)
    return Geometry.from_lines([l1, l2])


@pytest.fixture
def grid_geom() -> Geometry:
    """A small grid geometry via public API (ensures registry wiring works)."""
    from api import G

    return G.grid(subdivisions=(0.2, 0.2))


# ---- Digest control fixtures ------------------------------------------------


@pytest.fixture
def env_digest_off(monkeypatch: pytest.MonkeyPatch):
    """Disable Geometry.digest via env and restore after."""
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    yield
    monkeypatch.delenv("PXD_DISABLE_GEOMETRY_DIGEST", raising=False)


@pytest.fixture
def env_digest_on(monkeypatch: pytest.MonkeyPatch):
    """Ensure Geometry.digest is enabled (default)."""
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "0")
    yield
    monkeypatch.delenv("PXD_DISABLE_GEOMETRY_DIGEST", raising=False)


@pytest.fixture
def digest_hex() -> Callable[[Geometry], str]:
    """Return a function that yields a hex digest for a Geometry.

    - If Geometry.digest is available, use it.
    - If disabled via env, compute a fallback digest from arrays to keep tests working.
    """

    def _from_arrays(g: Geometry) -> bytes:
        c, o = g.as_arrays(copy=False)
        c = np.ascontiguousarray(c).view(np.uint8)
        o = np.ascontiguousarray(o).view(np.uint8)
        h = hashlib.blake2b(digest_size=16)
        h.update(c.tobytes())
        h.update(o.tobytes())
        return h.digest()

    def _digest_hex(g: Geometry) -> str:
        try:
            d = g.digest  # type: ignore[attr-defined]
        except Exception:
            d = _from_arrays(g)
        return d.hex()

    return _digest_hex


# ---- CC fixture for smoke tests --------------------------------------------


@pytest.fixture
def cc_min() -> Mapping[int, float]:
    """Minimal CC mapping expected by main.draw.

    Defaults: keys 1..7 -> 0.0, key 8 -> 1.0, key 9 -> 1.0
    """

    base: Dict[int, float] = {i: 0.0 for i in range(1, 10)}
    base[8] = 1.0
    base[9] = 1.0
    return base


# ---- Shape registry sanity --------------------------------------------------


@pytest.fixture
def shapes_exist() -> list[str]:
    """Return a list of registered shape names (non-empty)."""
    from api.shape_registry import list_registered_shapes

    names = list_registered_shapes()
    assert len(names) > 0, "No shapes registered; check shapes.registry wiring"
    return names


# ---- Snapshot helper fixture ------------------------------------------------


@pytest.fixture
def snapshot(tmp_path_factory):  # type: ignore[no-untyped-def]
    """Fixture that asserts or updates digest snapshots for the calling test.

    Usage:
        def test_something(snapshot, digest_hex, tiny_geom):
            snapshot([digest_hex(tiny_geom)])
    """
    import inspect

    from tests._utils import snapshot as _snap

    # Discover caller's module and function name
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    caller = frame.f_back
    func_name = caller.f_code.co_name
    module_name = caller.f_globals.get("__name__", "unknown").split(".")[-1]

    def _snapshot(lines: list[str], *, name: str | None = None) -> None:
        tname = name or func_name
        path = _snap.path_for(module_name, tname)
        _snap.assert_or_update(path, lines)

    return _snapshot
