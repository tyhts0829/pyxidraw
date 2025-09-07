from __future__ import annotations

"""Install minimal dummy modules for optional heavy deps used in tests/tools.

Provides compatible shims for:
- numba.njit: no-op decorator
- fontTools: minimal pens/ttLib used by shapes/text
- shapely: minimal geometry classes imported by effects/offset

This keeps tests and stub generation runnable in lean environments.
"""

import sys
import types


def install() -> None:
    # numba
    try:  # pragma: no cover - import path
        import numba  # noqa: F401
    except Exception:  # pragma: no cover - shim path
        m = types.ModuleType("numba")

        def _njit(*_a, **_k):
            def deco(fn):
                return fn

            return deco

        m.njit = _njit  # type: ignore[attr-defined]
        sys.modules["numba"] = m

    # fontTools
    try:  # pragma: no cover
        import fontTools  # noqa: F401
    except Exception:  # pragma: no cover
        ft = types.ModuleType("fontTools")
        pens = types.ModuleType("fontTools.pens")
        rec = types.ModuleType("fontTools.pens.recordingPen")

        class RecordingPen:  # pragma: no cover - dummy
            def __init__(self, *a, **k):
                pass

        rec.RecordingPen = RecordingPen
        ttLib = types.ModuleType("fontTools.ttLib")

        class TTFont:  # pragma: no cover - dummy
            def __init__(self, *a, **k):
                pass

            def __getitem__(self, key):
                return types.SimpleNamespace(unitsPerEm=1000)

        ttLib.TTFont = TTFont
        sys.modules["fontTools"] = ft
        sys.modules["fontTools.pens"] = pens
        sys.modules["fontTools.pens.recordingPen"] = rec
        sys.modules["fontTools.ttLib"] = ttLib

    # shapely
    try:  # pragma: no cover
        import shapely  # noqa: F401
    except Exception:  # pragma: no cover
        shp = types.ModuleType("shapely")
        geom = types.ModuleType("shapely.geometry")
        geom_base = types.ModuleType("shapely.geometry.base")

        class BaseGeometry:  # pragma: no cover - dummy
            pass

        class _G(BaseGeometry):  # pragma: no cover - dummy
            def __init__(self, *a, **k):
                self._coords = []

            @property
            def is_empty(self):
                return True

            @property
            def geoms(self):
                return []

            @property
            def coords(self):
                return self._coords

            def buffer(self, *a, **k):
                return self

        class LineString(_G):
            pass

        class MultiLineString(_G):
            pass

        class Polygon(_G):
            def __init__(self, *a, **k):
                super().__init__()
                self.exterior = types.SimpleNamespace(coords=[])

        class MultiPolygon(_G):
            pass

        geom_base.BaseGeometry = BaseGeometry
        geom.LineString = LineString
        geom.MultiLineString = MultiLineString
        geom.Polygon = Polygon
        geom.MultiPolygon = MultiPolygon
        sys.modules["shapely"] = shp
        sys.modules["shapely.geometry"] = geom
        sys.modules["shapely.geometry.base"] = geom_base
