"""
どこで: `engine.runtime` の結果コンテナ。
何を: ワーカからメインへ返す `RenderPacket`（`Geometry` と frame_id）。
なぜ: 描画対象と順序を明示し、受信側が最新フレームか判断できるようにするため。
"""

from dataclasses import dataclass
from typing import Mapping

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import Layer


@dataclass(slots=True, frozen=True)
class RenderPacket:
    """ワーカ → メインスレッドへ渡す描画データのコンテナ。"""

    geometry: Geometry | LazyGeometry | None
    frame_id: int  # ワーカ側で連番付与
    # 追加メトリクス（任意）: HUD 用にキャッシュの HIT/MISS を伝える。
    cache_flags: Mapping[str, str] | None = None  # keys: "shape"/"effect" → "HIT" or "MISS"
    # レイヤー描画がある場合はこちらを使用（geometry は None を推奨）。
    layers: tuple[Layer, ...] | None = None
