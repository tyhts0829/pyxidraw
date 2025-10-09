"""
どこで: `engine.runtime` のタスク定義。
何を: ワーカへ渡す 1 フレームぶんの `RenderTask`（時刻と CC 状態）。
なぜ: 実行キューの型を固定し、プロセス間/スレッド間での受け渡しを簡潔にするため。
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class RenderTask:
    """メインスレッド → ワーカへ送る描画タスク。"""

    frame_id: int
    t: float
    cc_state: Mapping[int, float]  # {CC#: normalized value 0.0-1.0}
    # Parameter GUI の override スナップショット（省略時 None）。
    # 形式は "{scope}.{name}#{index}.{param}" → 実値。
    param_overrides: Mapping[str, Any] | None = None
