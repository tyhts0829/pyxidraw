from __future__ import annotations

import math

import pytest


@pytest.mark.smoke
def test_import_and_basic_call() -> None:
    from api import lfo

    osc = lfo(wave="sine", freq=1.0)
    v = osc(0.25)
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "wave, kwargs",
    [
        ("sine", {}),
        ("triangle", {}),
        ("saw_up", {}),
        ("saw_down", {}),
        ("square", {}),
        ("pulse", {"pw": 0.3}),
        ("sh", {"seed": 123}),
        ("perlin", {"seed": 42, "octaves": 3, "persistence": 0.5, "lacunarity": 2.0}),
    ],
)
def test_value_range_0_1(wave: str, kwargs: dict[str, object]) -> None:
    from api import lfo

    osc = lfo(wave=wave, freq=1.7, **kwargs)
    for i in range(100):
        t = i * 0.01
        v = osc(t)
        assert 0.0 <= v <= 1.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "wave, kwargs",
    [
        ("sine", {}),
        ("triangle", {}),
        ("saw_up", {}),
        ("saw_down", {}),
        ("square", {}),
        ("pulse", {"pw": 0.3}),
    ],
)
def test_periodicity(wave: str, kwargs: dict[str, object]) -> None:
    from api import lfo

    freq = 1.3
    period = 1.0 / freq
    osc = lfo(wave=wave, freq=freq, **kwargs)
    for i in range(20):
        t = i * 0.07
        v1 = osc(t)
        v2 = osc(t + period)
        assert math.isclose(v1, v2, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.smoke
def test_pulse_pw_effect_monotonic() -> None:
    from api import lfo

    # phi=0.6 -> pw=0.3 では 0、pw=0.7 では 1
    osc_a = lfo(wave="pulse", freq=1.0, pw=0.3)
    osc_b = lfo(wave="pulse", freq=1.0, pw=0.7)
    t = 0.6
    assert osc_a(t) < osc_b(t)


@pytest.mark.smoke
def test_saw_skew_effect_direction() -> None:
    from api import lfo

    # skew を増やすと（gamma が増大）早期の値は小さくなる（saw_up）
    osc_lo = lfo(wave="saw_up", freq=1.0, skew=-1.0)
    osc_hi = lfo(wave="saw_up", freq=1.0, skew=1.0)
    t = 0.25
    assert osc_lo(t) > osc_hi(t)


@pytest.mark.smoke
def test_determinism_sh_and_perlin() -> None:
    from api import lfo

    sh1 = lfo(wave="sh", freq=1.0, seed=123)
    sh2 = lfo(wave="sh", freq=1.0, seed=123)
    sh3 = lfo(wave="sh", freq=1.0, seed=456)
    per1 = lfo(wave="perlin", freq=0.5, seed=999)
    per2 = lfo(wave="perlin", freq=0.5, seed=999)
    per3 = lfo(wave="perlin", freq=0.5, seed=1000)

    ts = [i * 0.07 for i in range(15)]
    assert all(math.isclose(sh1(t), sh2(t), abs_tol=1e-9) for t in ts)
    assert any(abs(sh1(t) - sh3(t)) > 1e-9 for t in ts)
    assert all(math.isclose(per1(t), per2(t), abs_tol=1e-12) for t in ts)
    assert any(abs(per1(t) - per3(t)) > 1e-9 for t in ts)


@pytest.mark.smoke
def test_perlin_continuity_small_dt() -> None:
    from api import lfo

    osc = lfo(wave="perlin", freq=0.5, seed=42)
    for i in range(10):
        t = i * 0.2
        dt = 1e-3
        dv = abs(osc(t + dt) - osc(t))
        assert dv < 0.05  # 小さい連続変化


@pytest.mark.smoke
def test_invalid_args_raise() -> None:
    from api import lfo

    with pytest.raises(ValueError):
        lfo(hi=0.0, lo=1.0)
    with pytest.raises(ValueError):
        lfo(freq=0.0)
    with pytest.raises(ValueError):
        lfo(period=0.0)
    with pytest.raises(ValueError):
        lfo(wave="pulse", pw=0.0)
    with pytest.raises(ValueError):
        lfo(wave="perlin", octaves=0)
    with pytest.raises(ValueError):
        lfo(wave="perlin", lacunarity=0.5)
    with pytest.raises(ValueError):
        lfo(wave="perlin", persistence=1.1)
