"""
どこで: `effects` 層。
何を: 線の色（RGB 0..1）と見かけの太さ倍率（1.0–10.0）を指示する非幾何エフェクト `style`。
なぜ: レンダラーにレイヤー単位の描画指示を与え、呼び出し単位で色/太さを指定可能にするため。

注意:
- 幾何（頂点/オフセット）自体は変更しない。関数は no-op で `Geometry` をそのまま返す。
- 実際の適用はランタイム側（Worker→Renderer）で `StyledLayer` 化して行う。
"""

from __future__ import annotations

from typing import Iterable

from effects.registry import effect
from engine.core.geometry import Geometry

PARAM_META = {
    "color": {
        "type": "vec3",
        "min": (0.0, 0.0, 0.0),
        "max": (1.0, 1.0, 1.0),
        # 0-255 UI と揃えた量子化（キャッシュ署名向け）
        "step": (1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
    },
    "thickness": {"type": "number", "min": 1.0, "max": 10.0, "step": 0.01},
}


@effect()
def style(
    g: Geometry,
    *,
    color: tuple[float, float, float] | Iterable[float] | None = None,
    thickness: float = 1.0,
) -> Geometry:
    """線の色（RGB 0..1）と太さ倍率を指定する（no-op）。

    挙動:
    - 幾何は変更しない（`g` をそのまま返す）。
    - 1 パイプライン内に複数指定があれば「後勝ち」（ランタイムで解決）。
    - Parameter GUI は `color` を RGB カラーピッカー（0–255、α非表示）、`thickness` を 1.0–10.0 の小数スライダーで編集。
    """

    return g


style.__param_meta__ = PARAM_META

# ランタイム検出用のマーカー
style.__effect_kind__ = "style"

__all__ = ["style"]
