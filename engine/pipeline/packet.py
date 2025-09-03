import time
from dataclasses import dataclass

from engine.core.geometry import Geometry


@dataclass(slots=True, frozen=True)
class RenderPacket:
    """ワーカ → メインスレッドへ渡す描画データのコンテナ。"""

    geometry: Geometry
    frame_id: int  # ワーカ側で連番付与
    timestamp: float = time.time()
