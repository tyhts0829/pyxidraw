"""
effects パッケージ（関数ベース）

このモジュールの import 副作用で関数エフェクトを登録します。
"""

from .registry import effect, get_effect, list_effects

# 関数エフェクトを登録（必要最小限）
from . import translate  # noqa: F401
from . import rotate  # noqa: F401
from . import scale  # noqa: F401
from . import displace  # noqa: F401
from . import fill  # noqa: F401
from . import repeat  # noqa: F401
from . import subdivide  # noqa: F401
from . import offset  # noqa: F401
from . import affine  # noqa: F401
from . import extrude  # noqa: F401
from . import boldify  # noqa: F401
from . import collapse  # noqa: F401
from . import dash  # noqa: F401
from . import ripple  # noqa: F401
from . import wobble  # noqa: F401
from . import explode  # noqa: F401
from . import twist  # noqa: F401
from . import trim  # noqa: F401
from . import weave  # noqa: F401

__all__ = [
    "effect",
    "get_effect",
    "list_effects",
]
