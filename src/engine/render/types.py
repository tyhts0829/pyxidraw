"""
どこで: `engine.render` 型定義。
何を: レイヤー描画用の軽量データクラス `Layer`。
なぜ: 1 フレーム内で色/太さが異なる複数のジオメトリを順描画するためのコンテナが必要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry


RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class Layer:
    """色/太さ・任意メタ付きの描画レイヤー。"""

    geometry: Geometry | LazyGeometry
    color: RGBA | None  # None なら Renderer の現在値を使用
    thickness: float | None  # None なら Renderer の現在値を使用
    name: str | None = None
    meta: dict[str, Any] | None = None


__all__ = ["Layer", "RGBA"]
