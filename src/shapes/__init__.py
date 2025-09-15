"""shapes パッケージ

破壊的変更により、shape は「関数の登録」のみを行います。
このモジュールはビルトイン shape の登録を副作用として行います。

現時点では関数化済みの shape を順次登録します（段階移行中）。
"""

# 関数版 shape 定義を import して登録（副作用）
from . import asemic_glyph as _register_asemic_glyph  # noqa: F401
from . import attractor as _register_attractor  # noqa: F401
from . import capsule as _register_capsule  # noqa: F401
from . import cone as _register_cone  # noqa: F401
from . import cylinder as _register_cylinder  # noqa: F401
from . import grid as _register_grid  # noqa: F401
from . import lissajous as _register_lissajous  # noqa: F401
from . import polygon as _register_polygon  # noqa: F401
from . import polyhedron as _register_polyhedron  # noqa: F401
from . import sphere as _register_sphere  # noqa: F401
from . import text as _register_text  # noqa: F401
from . import torus as _register_torus  # noqa: F401
from .registry import get_shape, is_shape_registered, list_shapes, shape  # re-export

__all__ = [
    "shape",
    "get_shape",
    "list_shapes",
    "is_shape_registered",
]
