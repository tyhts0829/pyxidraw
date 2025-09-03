import logging
from common.logging import setup_default_logging
from api import E, G


def build_geometry(t: float, cc: dict) -> "Geometry":
    # 形状の生成（Geometryを直接返す）
    sphere = G.sphere(subdivisions=cc.get(1, 0.5), sphere_type=cc.get(4, 0.5))
    sphere = sphere.scale(80, 80, 80).translate(100, 100, 0).rotate(
        x=cc.get(3, 0.1), y=cc.get(3, 0.1), z=cc.get(3, 0.1), center=(100, 100, 0)
    )

    # 関数パイプライン
    pipeline = (
        E.pipeline
        .noise(intensity=cc.get(2, 0.3), time=t)
        .rotation(center=(100, 100, 0), rotate=(0.1, 0.1, 0.1))
        .build()
    )
    return pipeline(sphere)


if __name__ == "__main__":
    # 簡易実行（依存のないスタブ）
    setup_default_logging()
    logger = logging.getLogger(__name__)
    g = build_geometry(0.0, {})
    c, o = g.as_arrays()
    logger.info("simple: points=%d, lines=%d", c.shape[0], max(0, o.shape[0]-1))
