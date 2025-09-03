"""
effects パッケージ（関数ベース）

このモジュールの import 副作用で関数エフェクトを登録します。
"""

from .registry import effect, get_effect, list_effects

# 関数エフェクトを登録（必要最小限）
from . import translation  # noqa: F401
from . import rotation  # noqa: F401
from . import scaling  # noqa: F401
from . import noise  # noqa: F401
from . import filling  # noqa: F401
from . import array  # noqa: F401
from . import subdivision  # noqa: F401
from . import buffer  # noqa: F401
from . import transform  # noqa: F401
from . import extrude  # noqa: F401
from . import boldify  # noqa: F401
from . import collapse  # noqa: F401
from . import dashify  # noqa: F401
from . import wave  # noqa: F401
from . import wobble  # noqa: F401
from . import explode  # noqa: F401
from . import twist  # noqa: F401
from . import trimming  # noqa: F401
from . import webify  # noqa: F401

__all__ = [
    "effect",
    "get_effect",
    "list_effects",
]
