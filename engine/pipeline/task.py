from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True, frozen=True)
class RenderTask:
    """メインスレッド → ワーカへ送る描画タスク。"""

    frame_id: int
    t: float
    cc_state: Mapping[int, int]  # {CC#: value}
