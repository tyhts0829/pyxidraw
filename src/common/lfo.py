"""
どこで: `common.lfo`
何を: 時間 t [秒] を入力として 0.0〜1.0 の値を返す LFO（低周波オシレータ）の純粋ロジックを提供。
なぜ: スケッチ/エフェクト内で任意パラメータの時間変調に用いるため。エンジン/IOに非依存の再利用可能部品。

設計方針:
- 純粋・決定的。副作用なし。型注釈あり。
- 波形: sine/triangle/saw_up/saw_down/square/pulse/sh/perlin。
- 範囲: 0..1 を既定とし、`lo..hi` へ線形射影の上で最終 clamp。
- 周波数/周期: `period` 指定時は優先（freq=1/period）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _frac(x: float) -> float:
    """x の小数部（0.0 <= r < 1.0）。負の値にも安定。"""
    r = x - math.floor(x)
    # 妥当化（ULP 誤差に対するガード）
    if r < 0.0:
        return 0.0
    if r >= 1.0:
        return 0.0
    return r


def _apply_skew(phi: float, skew: float) -> float:
    """位相 `phi` (0..1) に歪み `skew` (-1..1) を適用して返す。

    `skew=0` で恒等。`skew` が正で序盤を圧縮、負で終盤を圧縮する単調写像。
    実装は指数曲線 `phi**gamma`（gamma∈[0.25,4.0]）で近似。
    """
    s = max(-1.0, min(1.0, float(skew)))
    gamma = 0.25 * (1.0 - (s + 1.0) / 2.0) + 4.0 * ((s + 1.0) / 2.0)
    # gamma<1 でも >0 を保証
    gamma = max(1e-6, gamma)
    # 対称性を良くするため、gamma<1 の場合は 1-(1-phi)**(1/gamma) でも近いが
    # ここでは単純に phi**gamma とし、設計仕様（近似）に合わせる。
    return float(phi**gamma)


def _triangle_bipolar(phi: float) -> float:
    """三角波（-1..1）。phi∈[0,1)。"""
    if phi < 0.5:
        return 4.0 * phi - 1.0
    else:
        return 3.0 - 4.0 * phi


def _saw_up_bipolar(phi: float) -> float:
    return 2.0 * phi - 1.0


def _saw_down_bipolar(phi: float) -> float:
    return 1.0 - 2.0 * phi


def _square_bipolar(phi: float, threshold: float) -> float:
    return 1.0 if phi < threshold else -1.0


# ---- S&H: 決定的擬似乱数 -----------------------------------------------


def _splitmix64(x: int) -> int:
    """SplitMix64 由来の簡易 64bit ミキサ（決定的ランダム化）。"""
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def _rand01_from_seed_index(seed: int, idx: int) -> float:
    """種 `seed` とインデックス `idx` から [0,1) の一様乱数を決定的生成。"""
    x = _splitmix64((seed & 0xFFFFFFFFFFFFFFFF) ^ (idx & 0xFFFFFFFFFFFFFFFF))
    # 上位 53bit を IEEE754 の仮数として 0..1 に正規化
    mant = x >> 11  # 64-53=11
    return (mant & ((1 << 53) - 1)) / float(1 << 53)


# ---- Perlin 1D: 勾配ノイズ + fBm ---------------------------------------


def _fade(u: float) -> float:
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


@dataclass
class _PermTable:
    p: list[int]


def _lcg_step(x: int) -> int:
    return (6364136223846793005 * x + 1) & 0xFFFFFFFFFFFFFFFF


def _build_perm(seed: int) -> _PermTable:
    # Fisher–Yates による決定的シャッフル（独自 LCG）
    arr = list(range(256))
    st = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(255, 0, -1):
        st = _lcg_step(st)
        j = int(st % (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    # 参照容易化のため 512 に拡張
    return _PermTable(p=arr + arr)


def _grad1(hash_v: int, x: float) -> float:
    # 1D の勾配は ±1 とする
    g = 1.0 if (hash_v & 1) == 0 else -1.0
    return g * x


def _perlin1(x: float, pt: _PermTable) -> float:
    # 格子点
    xi = int(math.floor(x)) & 255
    xf = x - math.floor(x)
    u = _fade(xf)
    a = pt.p[xi]
    b = pt.p[xi + 1]
    # 左右の勾配値
    g0 = _grad1(a, xf)
    g1 = _grad1(b, xf - 1.0)
    # 線形補間
    return g0 + u * (g1 - g0)


def _perlin_fbm(
    x: float, *, seed: int, octaves: int, persistence: float, lacunarity: float
) -> float:
    pt = _build_perm(seed)
    o = max(1, int(octaves))
    p = max(0.0, min(1.0, float(persistence)))
    l = max(1.0, float(lacunarity))
    amp = 1.0
    freq = 1.0
    total = 0.0
    amp_sum = 0.0
    for _ in range(o):
        total += _perlin1(x * freq, pt) * amp
        amp_sum += amp
        amp *= p
        freq *= l
    if amp_sum <= 1e-12:
        return 0.0
    # [-1,1] へ正規化
    y = total / amp_sum
    # 念のための clamp（極端な境界）
    return max(-1.0, min(1.0, y))


# ---- 公開 API -----------------------------------------------------------


class LFO:
    """LFO（低周波オシレータ）。`__call__(t)` で 0..1 を返す。

    引数:
        wave: 波形種別（"sine"/"triangle"/"saw_up"/"saw_down"/"square"/"pulse"/"sh"/"perlin"）。
        freq: 周波数 [Hz]。`period` 指定時は無視される。
        period: 周期 [秒]。指定時は `freq = 1/period`。
        phase: 位相 [周期単位]。周期波形は wrap、Perlin は時間オフセット。
        lo: 出力下限。
        hi: 出力上限。
        pw: パルス幅（pulse のみ）。
        skew: 歪み（-1..1）。三角/ノコギリに適用。
        seed: 決定的シード（SH/Perlin）。None で固定既定。
        octaves: Perlin のオクターブ数（>=1）。
        persistence: Perlin の振幅減衰（0..1）。
        lacunarity: Perlin の周波数倍率（>=1）。
    """

    __slots__ = (
        "_wave",
        "_freq",
        "_phase",
        "_lo",
        "_hi",
        "_pw",
        "_skew",
        "_seed",
        "_octaves",
        "_persistence",
        "_lacunarity",
    )

    def __init__(
        self,
        *,
        wave: str = "sine",
        freq: float | None = 1.0,
        period: float | None = None,
        phase: float = 0.0,
        lo: float = 0.0,
        hi: float = 1.0,
        pw: float = 0.5,
        skew: float = 0.0,
        seed: int | None = None,
        octaves: int = 3,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> None:
        w = (wave or "sine").lower()
        if hi <= lo:
            raise ValueError("hi は lo より大きい必要がある")
        if period is not None:
            if period <= 0.0:
                raise ValueError("period は正の値が必要")
            f = 1.0 / float(period)
        else:
            f = float(freq if freq is not None else 1.0)
            if f <= 0.0:
                raise ValueError("freq は正の値が必要")
        if w in ("pulse",):
            if not (0.0 < pw < 1.0):
                raise ValueError("pw は (0,1) の範囲が必要")
        if w in ("perlin",):
            if octaves < 1:
                raise ValueError("octaves は 1 以上が必要")
            if lacunarity < 1.0:
                raise ValueError("lacunarity は 1.0 以上が必要")
            if not (0.0 <= persistence <= 1.0):
                raise ValueError("persistence は 0..1 が必要")

        self._wave = w
        self._freq = f
        self._phase = float(phase)
        self._lo = float(lo)
        self._hi = float(hi)
        self._pw = float(pw)
        self._skew = float(skew)
        self._seed = 0 if seed is None else int(seed)
        self._octaves = int(octaves)
        self._persistence = float(persistence)
        self._lacunarity = float(lacunarity)

    # ---- 内部ユーティリティ -------------------------------------------
    def _eval_bipolar(self, t: float) -> float:
        w = self._wave
        if w == "perlin":
            x = self._freq * float(t) + self._phase
            return _perlin_fbm(
                x,
                seed=self._seed,
                octaves=self._octaves,
                persistence=self._persistence,
                lacunarity=self._lacunarity,
            )

        # 周期波形
        phi = _frac(self._freq * float(t) + self._phase)
        if w == "sine":
            return math.sin(2.0 * math.pi * phi)
        if w == "triangle":
            phi_s = _apply_skew(phi, self._skew)
            return _triangle_bipolar(phi_s)
        if w == "saw_up":
            phi_s = _apply_skew(phi, self._skew)
            return _saw_up_bipolar(phi_s)
        if w == "saw_down":
            phi_s = _apply_skew(phi, self._skew)
            return _saw_down_bipolar(phi_s)
        if w == "square":
            return _square_bipolar(phi, 0.5)
        if w == "pulse":
            return _square_bipolar(phi, self._pw)
        if w == "sh":
            # 区間インデックス（phase は周期単位としてそのまま加算）
            k = int(math.floor(self._freq * float(t) + self._phase))
            u = _rand01_from_seed_index(self._seed, k)
            return 2.0 * u - 1.0
        # 既定: sine
        return math.sin(2.0 * math.pi * phi)

    # ---- 呼び出し -------------------------------------------------------
    def __call__(self, t: float) -> float:
        """時刻 `t` [秒] を評価して 0..1 の値を返す。"""
        y = self._eval_bipolar(float(t))  # [-1,1]
        n = (y + 1.0) * 0.5  # 0..1
        out = self._lo + (self._hi - self._lo) * n
        # 最終 clamp（数値誤差ガード）
        if out < self._lo:
            return self._lo
        if out > self._hi:
            return self._hi
        return out


def lfo(
    wave: str = "sine",
    *,
    freq: float | None = 1.0,
    period: float | None = None,
    phase: float = 0.0,
    lo: float = 0.0,
    hi: float = 1.0,
    pw: float = 0.5,
    skew: float = 0.0,
    seed: int | None = None,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> LFO:
    """LFO を構成して返すファクトリ。

    引数:
        wave: 波形（"sine"/"triangle"/"saw_up"/"saw_down"/"square"/"pulse"/"sh"/"perlin"）。
        freq: 周波数 [Hz]。`period` 指定時は無視。
        period: 周期 [秒]。指定時は `freq = 1/period`。
        phase: 位相 [周期単位]。Perlin は時間オフセット。
        lo, hi: 出力範囲。
        pw: パルス幅（pulse）。
        skew: 歪み（-1..1）。
        seed: 決定的シード（SH/Perlin）。
        octaves, persistence, lacunarity: Perlin fBm パラメータ。

    返り値:
        LFO: `__call__(t: float) -> float` を持つ呼び出し可能。
    """
    return LFO(
        wave=wave,
        freq=freq,
        period=period,
        phase=phase,
        lo=lo,
        hi=hi,
        pw=pw,
        skew=skew,
        seed=seed,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
    )


__all__ = ["LFO", "lfo"]
