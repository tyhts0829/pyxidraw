"""
どこで: `effects` パッケージ（関数ベース）。
何を: Geometry→Geometry の純関数エフェクトを登録し、`api` から利用可能にする。
なぜ: 生成/加工/描画の責務分離に従い、加工ステージの拡張点を一箇所に集約するため。
"""

# 関数エフェクトを登録（必要最小限）
from . import affine  # noqa: F401
from . import boldify  # noqa: F401
from . import collapse  # noqa: F401
from . import dash  # noqa: F401
from . import displace  # noqa: F401
from . import explode  # noqa: F401
from . import extrude  # noqa: F401
from . import fill  # noqa: F401
from . import mirror  # noqa: F401
from . import mirror3d  # noqa: F401
from . import offset  # noqa: F401
from . import partition  # noqa: F401
from . import repeat  # noqa: F401
from . import rotate  # noqa: F401
from . import scale  # noqa: F401
from . import subdivide  # noqa: F401
from . import translate  # noqa: F401
from . import trim  # noqa: F401
from . import twist  # noqa: F401
from . import weave  # noqa: F401
from . import wobble  # noqa: F401
from .registry import effect, get_effect, list_effects

__all__ = [
    "effect",
    "get_effect",
    "list_effects",
]
