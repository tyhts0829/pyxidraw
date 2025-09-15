import numpy as np
import pytest


@pytest.mark.smoke
def test_api_effect_import_and_pipeline():
    from api import E, effect
    from engine.core.geometry import Geometry

    NAME = "tmp_fx_smoke"

    # 登録
    @effect(NAME)
    def _fx(g: Geometry, *, dx: float = 0.0) -> Geometry:
        # x 平行移動（簡易に実装）
        c, o = g.as_arrays(copy=True)
        c[:, 0] = c[:, 0] + float(dx)
        out = Geometry.from_lines([c[o[i] : o[i + 1]] for i in range(len(o) - 1)])
        return out

    # 最小のジオメトリ
    xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([xy])

    pipe = E.pipeline.tmp_fx_smoke(dx=2.5).build()
    out = pipe(g)

    cc, _ = out.as_arrays(copy=False)
    assert np.allclose(cc[:2, 0], [2.5, 3.5], atol=1e-5)
    # 後片付け（レジストリから削除してスタブ生成への影響を避ける）
    from effects import registry as _ereg

    _ereg.unregister(NAME)


def test_no_register_effect_alias_in_api():
    with pytest.raises(ImportError):
        from api import register_effect  # noqa: F401
