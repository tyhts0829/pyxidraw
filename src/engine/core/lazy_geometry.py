"""
どこで: `engine.core.lazy_geometry`
何を: 形状とエフェクトの“計画（spec/plan）”を保持し、必要時に `Geometry` へ実体化（realize）する軽量ラッパ。
なぜ: shape/effect を既定で遅延化し、終端で一括評価する設計にするため。

設計要点:
- `base` は shape の spec（関数参照とパラメータ）または実体 `Geometry`。
- `plan` は effect の列（関数参照とパラメータ）。順序は固定、途中の暗黙 realize は行わない。
- `realize()` は 1 回だけ評価して `Geometry` を返す（以降はキャッシュを返す）。
- 読み出し系（`as_arrays`/`len`/`is_empty`）は必要に応じて自動 `realize()`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any, Callable, Literal

from engine.core.geometry import Geometry


BaseKind = Literal["shape", "geometry"]


@dataclass
class LazyGeometry:
    base_kind: BaseKind
    base_payload: Any  # (shape_impl: Callable, params_dict) or Geometry
    plan: list[tuple[Callable[[Geometry], Geometry], dict[str, Any]]] = field(default_factory=list)

    _cached: Geometry | None = field(default=None, init=False, repr=False)

    # ---- 実体化 ---------------------------------------------------------
    def realize(self) -> Geometry:
        if self._cached is not None:
            return self._cached

        from common.param_utils import params_signature as _params_signature  # local import

        # 1) shape 実体化
        if self.base_kind == "geometry":
            g = self.base_payload
            assert isinstance(g, Geometry)
            base_key: Any = ("geom-id", id(g))
        else:
            shape_impl, params = self.base_payload
            # 形状結果の LRU キャッシュ（フレーム間共有）
            try:
                params_tuple = _params_signature(shape_impl, dict(params))
                impl_id = (_safe_getattr(shape_impl, "__module__"), _safe_name(shape_impl))
                key = ("shape", impl_id, params_tuple)
            except Exception:
                key = None  # フォールバック: キャッシュを使わない

            g = None
            if key is not None:
                try:
                    _entry = _SHAPE_CACHE.get(key)
                except Exception:
                    _entry = None
                if _entry is not None:
                    try:
                        # LRU: move to end
                        _ = _SHAPE_CACHE.pop(key)
                        _SHAPE_CACHE[key] = _entry
                    except Exception:
                        pass
                    g = _entry
            if g is None:
                # 実生成
                g = shape_impl(**params)  # ユーザー shape は Geometry または lines を返す前提
                if not isinstance(g, Geometry):
                    # 便宜: lines を許容
                    g = Geometry.from_lines(g)
                # キャッシュへ格納
                if key is not None:
                    try:
                        _SHAPE_CACHE[key] = g
                        if _SHAPE_CACHE_MAXSIZE is not None and _SHAPE_CACHE_MAXSIZE > 0:
                            while len(_SHAPE_CACHE) > _SHAPE_CACHE_MAXSIZE:
                                _SHAPE_CACHE.popitem(last=False)
                    except Exception:
                        pass
            base_key = key or ("shape", id(shape_impl))

        # 2) effect を順次適用（Prefix LRU による途中結果の再利用を試みる）
        out = g
        steps = list(self.plan)
        start_idx = 0

        # ---- Prefix Cache: 最長一致プレフィックスを検索 ----
        try:
            if _PREFIX_CACHE_ENABLED and steps:
                effect_sigs: list[tuple[tuple[str, str], tuple[tuple[str, object], ...]]] = []
                for impl, eparams in steps:
                    try:
                        e_tuple = _params_signature(impl, dict(eparams))
                    except Exception:
                        e_tuple = tuple()
                    effect_sigs.append(
                        ((_safe_getattr(impl, "__module__"), _safe_name(impl)), e_tuple)
                    )

                for i in range(len(effect_sigs), 0, -1):
                    k = (base_key, tuple(effect_sigs[:i]))
                    try:
                        cached_geo = _PREFIX_CACHE.get(k)
                    except Exception:
                        cached_geo = None
                    if cached_geo is not None:
                        # LRU: move to end
                        try:
                            _ = _PREFIX_CACHE.pop(k)
                            _PREFIX_CACHE[k] = cached_geo
                        except Exception:
                            pass
                        out = cached_geo
                        start_idx = i
                        _PREFIX_HITS_INC()
                        break
                else:
                    _PREFIX_MISSES_INC()
        except Exception:
            pass

        # ---- 残りの effect を適用 ----
        try:
            # 事前に効果署名列を用意（上と同等）
            try:
                effect_sigs_all = []
                for impl, eparams in steps:
                    e_tuple = _params_signature(impl, dict(eparams))
                    effect_sigs_all.append(
                        ((_safe_getattr(impl, "__module__"), _safe_name(impl)), e_tuple)
                    )
            except Exception:
                effect_sigs_all = []

            for i in range(start_idx, len(steps)):
                impl, params = steps[i]
                out = impl(out, **params)

                # プレフィックスキャッシュへの格納（最大頂点数などの制約つき）
                if _PREFIX_CACHE_ENABLED:
                    # 頂点数ガード
                    try:
                        vcount = int(out.coords.shape[0])
                    except Exception:
                        vcount = 0
                    k = (
                        (base_key, tuple(effect_sigs_all[: i + 1]))
                        if steps
                        else (base_key, tuple())
                    )
                    if _PREFIX_CACHE_MAXSIZE != 0 and vcount <= _PREFIX_CACHE_MAX_VERTS:
                        _PREFIX_CACHE[k] = out
                        # LRU 制御
                        if _PREFIX_CACHE_MAXSIZE is not None and _PREFIX_CACHE_MAXSIZE > 0:
                            while len(_PREFIX_CACHE) > _PREFIX_CACHE_MAXSIZE:
                                try:
                                    _PREFIX_CACHE.popitem(last=False)
                                    _PREFIX_EVICTS_INC()
                                except Exception:
                                    break
                        _PREFIX_STORES_INC()
        except Exception:
            pass

        self._cached = out
        return out

    # ---- 読み出し系 -----------------------------------------------------
    def as_arrays(self, *, copy: bool = False):
        g = self.realize()
        return g.as_arrays(copy=copy)

    @property
    def is_empty(self) -> bool:
        return self.realize().is_empty

    def __len__(self) -> int:
        c, _ = self.as_arrays(copy=False)
        return int(c.shape[0])

    # ---- 便宜的な幾何操作（エフェクトへ写像） -------------------------------
    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> "LazyGeometry":
        params = {"delta": (float(x), float(y), float(z))}
        return LazyGeometry(
            self.base_kind, self.base_payload, self.plan + [(_fx_translate, params)]
        )

    def scale(
        self,
        sx: float = 1.0,
        sy: float | None = None,
        sz: float | None = None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "LazyGeometry":
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        params = {
            "auto_center": False,
            "pivot": (float(center[0]), float(center[1]), float(center[2])),
            "scale": (float(sx), float(sy), float(sz)),
        }
        return LazyGeometry(self.base_kind, self.base_payload, self.plan + [(_fx_scale, params)])

    def rotate(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "LazyGeometry":
        params = {
            "auto_center": False,
            "pivot": (float(center[0]), float(center[1]), float(center[2])),
            "angles_rad": (float(x), float(y), float(z)),
        }
        return LazyGeometry(self.base_kind, self.base_payload, self.plan + [(_fx_rotate, params)])

    # Geometry 互換の便宜プロパティ
    @property
    def n_vertices(self) -> int:
        c, _ = self.as_arrays(copy=False)
        return int(c.shape[0])

    @property
    def n_lines(self) -> int:
        _, o = self.as_arrays(copy=False)
        return int(o.shape[0] - 1) if o.size > 0 else 0


# ---- 形状結果 LRU（プロセス内共有） -----------------------------------------
import os as _os  # noqa: E402  # 遅延でなくモジュール末尾にまとめて定義

try:
    _SHAPE_CACHE_MAXSIZE_ENV = _os.getenv("PXD_SHAPE_CACHE_MAXSIZE")
    _SHAPE_CACHE_MAXSIZE: int | None = (
        int(_SHAPE_CACHE_MAXSIZE_ENV) if _SHAPE_CACHE_MAXSIZE_ENV is not None else 128
    )
    if _SHAPE_CACHE_MAXSIZE is not None and _SHAPE_CACHE_MAXSIZE < 0:
        _SHAPE_CACHE_MAXSIZE = 0
except Exception:
    _SHAPE_CACHE_MAXSIZE = 128

_SHAPE_CACHE: "OrderedDict[object, Geometry]" = OrderedDict()

# ---- Prefix（途中結果）LRU --------------------------------------------------
try:
    _PREFIX_CACHE_ENABLED = int(_os.getenv("PXD_PREFIX_CACHE_ENABLED", "1")) != 0
except Exception:
    _PREFIX_CACHE_ENABLED = True
try:
    _PREFIX_CACHE_MAXSIZE_ENV = _os.getenv("PXD_PREFIX_CACHE_MAXSIZE")
    _PREFIX_CACHE_MAXSIZE: int | None = (
        int(_PREFIX_CACHE_MAXSIZE_ENV) if _PREFIX_CACHE_MAXSIZE_ENV is not None else 128
    )
    if _PREFIX_CACHE_MAXSIZE is not None and _PREFIX_CACHE_MAXSIZE < 0:
        _PREFIX_CACHE_MAXSIZE = 0
except Exception:
    _PREFIX_CACHE_MAXSIZE = 128
try:
    _PREFIX_CACHE_MAX_VERTS = int(_os.getenv("PXD_PREFIX_CACHE_MAX_VERTS", "10000000"))
except Exception:
    _PREFIX_CACHE_MAX_VERTS = 10_000_000
try:
    _PREFIX_DEBUG = int(_os.getenv("PXD_DEBUG_PREFIX_CACHE", "0")) != 0
except Exception:
    _PREFIX_DEBUG = False

_PREFIX_CACHE: "OrderedDict[object, Geometry]" = OrderedDict()
_PREFIX_HITS = 0
_PREFIX_MISSES = 0
_PREFIX_STORES = 0
_PREFIX_EVICTS = 0
_PREFIX_TAIL_SKIPPED = 0


def _PREFIX_HITS_INC() -> None:
    global _PREFIX_HITS
    _PREFIX_HITS += 1


def _PREFIX_MISSES_INC() -> None:
    global _PREFIX_MISSES
    _PREFIX_MISSES += 1


def _PREFIX_STORES_INC() -> None:
    global _PREFIX_STORES
    _PREFIX_STORES += 1


def _PREFIX_EVICTS_INC() -> None:
    global _PREFIX_EVICTS
    _PREFIX_EVICTS += 1


try:
    _PREFIX_STORE_TAIL = int(_os.getenv("PXD_PREFIX_CACHE_STORE_TAIL", "0")) != 0
except Exception:
    _PREFIX_STORE_TAIL = False
try:
    _PREFIX_STORE_ON_MISS_UP_TO = int(_os.getenv("PXD_PREFIX_CACHE_STORE_ON_MISS_UP_TO", "16"))
except Exception:
    _PREFIX_STORE_ON_MISS_UP_TO = 16


def _PREFIX_TAIL_SKIPPED_INC() -> None:
    global _PREFIX_TAIL_SKIPPED
    _PREFIX_TAIL_SKIPPED += 1


# ---- コア内蔵の軽量エフェクト（engine.core 依存のみ） ----------------------
def _fx_translate(g: Geometry, *, delta: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Geometry:
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    return g.translate(dx, dy, dz)


def _fx_scale(
    g: Geometry,
    *,
    auto_center: bool = False,
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Geometry:
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    cx, cy, cz = float(pivot[0]), float(pivot[1]), float(pivot[2])
    center = (cx, cy, cz)
    return g.scale(sx, sy, sz, center=center)


def _fx_rotate(
    g: Geometry,
    *,
    auto_center: bool = False,
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angles_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Geometry:
    rx, ry, rz = float(angles_rad[0]), float(angles_rad[1]), float(angles_rad[2])
    cx, cy, cz = float(pivot[0]), float(pivot[1]), float(pivot[2])
    center = (cx, cy, cz)
    return g.rotate(x=rx, y=ry, z=rz, center=center)


def _safe_getattr(obj: Any, name: str) -> str:
    try:
        v = getattr(obj, name)
        return str(v)
    except Exception:
        return ""


def _safe_name(fn: Callable[..., Any]) -> str:
    try:
        return str(getattr(fn, "__qualname__", getattr(fn, "__name__", "")))
    except Exception:
        return ""
