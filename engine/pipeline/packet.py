import time
from dataclasses import dataclass

from api.geometry_api import GeometryAPI


@dataclass(slots=True, frozen=True)
class RenderPacket:
    """ワーカ → メインスレッドへ渡す描画データのコンテナ。"""

    geometry: GeometryAPI
    frame_id: int  # ワーカ側で連番付与
    timestamp: float = time.time()
