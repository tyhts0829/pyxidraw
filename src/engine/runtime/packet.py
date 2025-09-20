"""
どこで: `engine.runtime` の結果コンテナ。
何を: ワーカからメインへ返す `RenderPacket`（`Geometry` と frame_id）。
なぜ: 描画対象と順序を明示し、受信側が最新フレームか判断できるようにするため。
"""

from dataclasses import dataclass

from engine.core.geometry import Geometry


@dataclass(slots=True, frozen=True)
class RenderPacket:
    """ワーカ → メインスレッドへ渡す描画データのコンテナ。"""

    geometry: Geometry
    frame_id: int  # ワーカ側で連番付与
