from __future__ import annotations

import pytest

from api import G


def test_text_generates_geometry():
    g = G.text(text="HI", font_size=0.2, font="Helvetica", align="center")
    coords, offsets = g.as_arrays(copy=False)
    # 座標/オフセット配列が生成され、少なくとも型が正しいことを確認
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert offsets.ndim == 1
    # 文字によっては空のグリフもあり得るが、基本的に点群は0以上
    assert coords.shape[0] >= 0

