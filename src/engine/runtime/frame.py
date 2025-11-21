"""
どこで: `engine.runtime` の描画ペイロード型。
何を: Renderer へ渡す 1 フレームぶんのデータ（Geometry または Layer 列）を表す。
なぜ: SwapBuffer/StreamReceiver/Renderer 間の契約を明示し、duck-typing を排除するため。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import Layer


@dataclass(frozen=True)
class RenderFrame:
    """描画対象 1 フレーム分のデータコンテナ。"""

    geometry: Geometry | LazyGeometry | None = None
    layers: tuple[Layer, ...] | None = None

    @property
    def has_layers(self) -> bool:
        return self.layers is not None and len(self.layers) > 0

    @classmethod
    def from_geometry(cls, geometry: Geometry | LazyGeometry | None) -> "RenderFrame":
        return cls(geometry=geometry, layers=None)

    @classmethod
    def from_layers(cls, layers: Sequence[Layer]) -> "RenderFrame":
        return cls(geometry=None, layers=tuple(layers))


__all__ = ["RenderFrame"]
