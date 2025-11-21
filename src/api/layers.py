"""
どこで: `api` 層のレイヤー組み立てヘルパ。
何を: Geometry/LazyGeometry を明示的なレイヤーとして束ねる API `L` を提供する。
なぜ: 描画レイヤーをユーザーコードで直接構築できるようにし、style エフェクトを介さずに色/太さを指定するため。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import Layer
from util.color import normalize_color as _norm_color


def _to_rgba(color: object | None) -> tuple[float, float, float, float] | None:
    if color is None:
        return None
    try:
        r, g, b, a = _norm_color(color)
        return float(r), float(g), float(b), float(a)
    except Exception:
        return None


def _make_layer(
    geometry: Geometry | LazyGeometry,
    *,
    color: object | None = None,
    thickness: float | None = None,
    name: str | None = None,
    meta: dict[str, object] | None = None,
) -> Layer:
    rgba = _to_rgba(color)
    return Layer(
        geometry=geometry,
        color=rgba,
        thickness=float(thickness) if thickness is not None else None,
        name=name,
        meta=meta,
    )


class LayerBuilder:
    """レイヤーを順番に積むための簡易ビルダー。"""

    def __init__(self) -> None:
        self._layers: list[Layer] = []

    def add(
        self,
        geometry: Geometry | LazyGeometry,
        *,
        color: object | None = None,
        thickness: float | None = None,
        name: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> "LayerBuilder":
        layer = _make_layer(
            geometry,
            color=color,
            thickness=thickness,
            name=name,
            meta=meta,
        )
        self._layers.append(layer)
        return self

    def build(self) -> tuple[Layer, ...]:
        return tuple(self._layers)


class _LayersAPI:
    """レイヤー構築用の公開 API。"""

    def layer(
        self,
        geometry: Geometry | LazyGeometry,
        *,
        color: object | None = None,
        thickness: float | None = None,
        name: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> Layer:
        """単一レイヤーを構築する。"""
        return _make_layer(
            geometry,
            color=color,
            thickness=thickness,
            name=name,
            meta=meta,
        )

    def of(
        self,
        geometries: Sequence[Geometry | LazyGeometry],
        *,
        color: object | None = None,
        thickness: float | None = None,
        name: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> tuple[Layer, ...]:
        """複数の Geometry/LazyGeometry から同一設定のレイヤーを生成する。"""
        return tuple(
            _make_layer(
                g,
                color=color,
                thickness=thickness,
                name=name,
                meta=meta,
            )
            for g in list(geometries)
        )

    def builder(self) -> LayerBuilder:
        """複数レイヤーを順に積むビルダーを返す。"""
        return LayerBuilder()

    def sequence(self, layers: Iterable[Layer]) -> tuple[Layer, ...]:
        """レイヤー列をタプル化して返す。"""
        return tuple(layers)


L = _LayersAPI()

__all__ = ["L", "Layer", "LayerBuilder"]
