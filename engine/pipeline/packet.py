import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class RenderPacket:
    """ワーカ → メインスレッドへ渡す描画データのコンテナ。"""

    vertices_list: Sequence[np.ndarray]
    frame_id: int  # ワーカ側で連番付与
    timestamp: float = time.time()
