from __future__ import annotations

import logging
import os
from typing import Mapping

import numpy as np

from api import E, G, run
from engine.core.geometry import Geometry


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    sphere = G.sphere(subdivisions=cc.get(1, 0.5), sphere_type=cc.get(2, 0.5))
    # 新API: rotate は angles_rad(Vec3) を受理
    pipe = E.pipeline.rotate(angles_rad=(0.0, 0.0, t)).build()
    return pipe(sphere.scale(100).translate(200, 200))


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(400, 400),
        render_scale=4,
        use_midi=False,
    )
