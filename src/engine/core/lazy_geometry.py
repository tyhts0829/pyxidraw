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

from engine.core.affine_ops import rotate as _fx_rotate
from engine.core.affine_ops import scale as _fx_scale
from engine.core.affine_ops import translate as _fx_translate
from engine.core.geometry import Geometry
from common.func_id import impl_id as _impl_id
from common.env import env_int, env_bool


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
                key = ("shape", _impl_id(shape_impl), params_tuple)
            except Exception:
                key = None  # フォールバック: キャッシュを使わない

            g = None
            if key is not None:
                _entry = _SHAPE_CACHE.get(key)
                if _entry is not None:
                    # LRU: move to end
                    _ = _SHAPE_CACHE.pop(key)
                    _SHAPE_CACHE[key] = _entry
                    g = _entry
            if g is None:
                # 実生成
                g = shape_impl(**params)  # ユーザー shape は Geometry または lines を返す前提
                if not isinstance(g, Geometry):
                    # 便宜: lines を許容
                    g = Geometry.from_lines(g)
                # キャッシュへ格納
                if key is not None:
                    _SHAPE_CACHE[key] = g
                    if _SHAPE_CACHE_MAXSIZE is not None and _SHAPE_CACHE_MAXSIZE > 0:
                        while len(_SHAPE_CACHE) > _SHAPE_CACHE_MAXSIZE:
                            _SHAPE_CACHE.popitem(last=False)
            base_key = key or ("shape", id(shape_impl))

        # 2) effect を順次適用（Prefix LRU による途中結果の再利用を試みる）
        out = g
        steps = list(self.plan)
        start_idx = 0

        # ---- Prefix Cache: 最長一致プレフィックスを検索 ----
        # 署名生成時に ObjectRef など実体参照を除外
        def _filter_sig_params(p: dict[str, Any]) -> dict[str, Any]:
            try:
                from common.types import ObjectRef as _ObjectRef  # local import
            except Exception:
                _ObjectRef = object  # type: ignore
            out: dict[str, Any] = {}
            for k, v in dict(p).items():
                if isinstance(k, str) and (k.endswith("_ref") or "_ref" in k):
                    continue
                if isinstance(v, _ObjectRef):
                    continue
                out[k] = v
            return out

        if _PREFIX_CACHE_ENABLED and steps:
            effect_sigs: list[tuple[str, tuple[tuple[str, object], ...]]] = []
            for impl, eparams in steps:
                try:
                    # 署名生成に失敗した場合は「非キャッシュ相当」として空署名を採用
                    e_tuple = _params_signature(impl, _filter_sig_params(dict(eparams)))
                except Exception:
                    e_tuple = tuple()
                effect_sigs.append((_impl_id(impl), e_tuple))

            for i in range(len(effect_sigs), 0, -1):
                k = (base_key, tuple(effect_sigs[:i]))
                cached_geo = _PREFIX_CACHE.get(k)
                if cached_geo is not None:
                    # LRU: move to end
                    _ = _PREFIX_CACHE.pop(k)
                    _PREFIX_CACHE[k] = cached_geo
                    out = cached_geo
                    start_idx = i
                    _PREFIX_HITS_INC()
                    break
            else:
                _PREFIX_MISSES_INC()

        # ---- 残りの effect を適用 ----
        # 事前に効果署名列を用意（上と同等）。署名生成に失敗した場合は保存キーに空署名を使用。
        try:
            effect_sigs_all = []
            for impl, eparams in steps:
                e_tuple = _params_signature(impl, _filter_sig_params(dict(eparams)))
                effect_sigs_all.append((_impl_id(impl), e_tuple))
        except Exception:
            effect_sigs_all = []

        for i in range(start_idx, len(steps)):
            impl, params = steps[i]
            out = impl(out, **params)

            # プレフィックスキャッシュへの格納（最大頂点数などの制約つき）
            if _PREFIX_CACHE_ENABLED:
                vcount = int(out.coords.shape[0])
                k = (base_key, tuple(effect_sigs_all[: i + 1])) if steps else (base_key, tuple())
                if _PREFIX_CACHE_MAXSIZE != 0 and vcount <= _PREFIX_CACHE_MAX_VERTS:
                    _PREFIX_CACHE[k] = out
                    # LRU 制御
                    if _PREFIX_CACHE_MAXSIZE is not None and _PREFIX_CACHE_MAXSIZE > 0:
                        while len(_PREFIX_CACHE) > _PREFIX_CACHE_MAXSIZE:
                            _PREFIX_CACHE.popitem(last=False)
                            _PREFIX_EVICTS_INC()
                    _PREFIX_STORES_INC()

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

    # ---- 演算子糖衣（Lazy 同士の連結のみ対応） ------------------------------
    def __add__(self, other: "LazyGeometry") -> "LazyGeometry":
        """糖衣: LazyGeometry 同士の結合を遅延で表現する。

        即時実体化は行わず、末尾に結合ステップ（_fx_concat_many）を集約して追加する。

        Notes
        -----
        - 右項が `LazyGeometry` 以外の場合は `NotImplemented` を返す（混在加算は非対応）。
        """
        if not isinstance(other, LazyGeometry):
            return NotImplemented

        from common.types import ObjectRef as _ObjectRef  # local import

        rhs_sig = _lazy_sig_for(other)
        rhs_ref = _ObjectRef(other)

        if self.plan and self.plan[-1][0] is _fx_concat_many:
            last_impl, last_params = self.plan[-1]
            rhs_sig_list = tuple(last_params.get("rhs_sig_list", ())) + (rhs_sig,)  # type: ignore[assignment]
            rhs_ref_list = list(last_params.get("rhs_ref_list", [])) + [rhs_ref]  # type: ignore[assignment]
            new_params = {"rhs_sig_list": rhs_sig_list, "rhs_ref_list": rhs_ref_list}
            new_plan = list(self.plan)
            new_plan[-1] = (last_impl, new_params)
            return LazyGeometry(self.base_kind, self.base_payload, new_plan)

        params = {"rhs_sig_list": (rhs_sig,), "rhs_ref_list": [rhs_ref]}
        return LazyGeometry(
            self.base_kind, self.base_payload, self.plan + [(_fx_concat_many, params)]
        )


# ---- 形状結果 LRU（プロセス内共有） -----------------------------------------

try:
    from common.settings import get as _get_settings

    _s = _get_settings()
    _SHAPE_CACHE_MAXSIZE: int | None = (
        int(_s.SHAPE_CACHE_MAXSIZE) if _s.SHAPE_CACHE_MAXSIZE is not None else None
    )
    _PREFIX_CACHE_ENABLED = bool(_s.PREFIX_CACHE_ENABLED)
    _PREFIX_CACHE_MAXSIZE: int | None = (
        int(_s.PREFIX_CACHE_MAXSIZE) if _s.PREFIX_CACHE_MAXSIZE is not None else None
    )
    _PREFIX_CACHE_MAX_VERTS: int = int(_s.PREFIX_CACHE_MAX_VERTS or 0)
    _PREFIX_DEBUG = bool(_s.DEBUG_PREFIX_CACHE)
except Exception:
    _SHAPE_CACHE_MAXSIZE = env_int("PXD_SHAPE_CACHE_MAXSIZE", 128, min_value=0)
    _PREFIX_CACHE_ENABLED = env_bool("PXD_PREFIX_CACHE_ENABLED", True)
    _PREFIX_CACHE_MAXSIZE = env_int("PXD_PREFIX_CACHE_MAXSIZE", 128, min_value=0)
    _PREFIX_CACHE_MAX_VERTS = env_int("PXD_PREFIX_CACHE_MAX_VERTS", 10_000_000, min_value=0) or 0
    _PREFIX_DEBUG = env_bool("PXD_DEBUG_PREFIX_CACHE", False)

_SHAPE_CACHE: "OrderedDict[object, Geometry]" = OrderedDict()

# ---- Prefix（途中結果）LRU --------------------------------------------------

_PREFIX_CACHE: "OrderedDict[object, Geometry]" = OrderedDict()
_PREFIX_HITS = 0
_PREFIX_MISSES = 0
_PREFIX_STORES = 0
_PREFIX_EVICTS = 0


def _PREFIX_HITS_INC() -> None:
    # デバッグ無効時はカウンタ更新をスキップ（実行コスト削減）
    if not _PREFIX_DEBUG:
        return
    global _PREFIX_HITS
    _PREFIX_HITS += 1


def _PREFIX_MISSES_INC() -> None:
    if not _PREFIX_DEBUG:
        return
    global _PREFIX_MISSES
    _PREFIX_MISSES += 1


def _PREFIX_STORES_INC() -> None:
    if not _PREFIX_DEBUG:
        return
    global _PREFIX_STORES
    _PREFIX_STORES += 1


def _PREFIX_EVICTS_INC() -> None:
    if not _PREFIX_DEBUG:
        return
    global _PREFIX_EVICTS
    _PREFIX_EVICTS += 1


## 末尾保存やミス時プリウォームの環境スイッチは未採用（将来案）。


def _fx_concat_many(
    g: Geometry,
    *,
    rhs_sig_list: tuple[object, ...] | tuple = tuple(),
    rhs_ref_list: list[object] | tuple = tuple(),
) -> Geometry:
    """Lazy 結合の終端: 複数の LazyGeometry を一括実体化して連結する。

    - 署名は `rhs_sig_list` のみ使用（`rhs_ref_list` はキャッシュキーから除外）。
    - 1 パスで coords/offsets を構築して O(totalN)。
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover - 環境依存
        # フォールバック: Geometry.concat 逐次（性能は低下するが安全）
        out = g
        for ref in rhs_ref_list:
            try:
                obj = getattr(ref, "unwrap", lambda: getattr(ref, "obj", ref))()
            except Exception:
                obj = getattr(ref, "obj", ref)
            if isinstance(obj, LazyGeometry):
                obj = obj.realize()
            if isinstance(obj, Geometry):
                out = out.concat(obj)
        return out

    # 正規化: 右項を Geometry にそろえる
    geoms: list[Geometry] = [g]
    for ref in list(rhs_ref_list):
        try:
            obj = getattr(ref, "unwrap", lambda: getattr(ref, "obj", ref))()
        except Exception:
            obj = getattr(ref, "obj", ref)
        if isinstance(obj, LazyGeometry):
            geoms.append(obj.realize())
        elif isinstance(obj, Geometry):
            geoms.append(obj)
        else:
            raise TypeError("rhs_ref_list には LazyGeometry/Geometry のみを指定してください")

    # 総量見積り
    total_n = 0
    total_m = 0
    for gg in geoms:
        total_n += int(gg.coords.shape[0])
        total_m += max(int(gg.offsets.shape[0]) - 1, 0)

    if total_m == 0:
        # すべて空線集合
        return Geometry(g.coords.copy(), g.offsets.copy())

    coords = np.empty((total_n, 3), dtype=np.float32)
    offsets = np.empty((total_m + 1,), dtype=np.int32)
    offsets[0] = 0
    v_cursor = 0
    o_cursor = 1
    for gg in geoms:
        n = int(gg.coords.shape[0])
        m = max(int(gg.offsets.shape[0]) - 1, 0)
        if n:
            coords[v_cursor : v_cursor + n] = gg.coords
        if m:
            offsets[o_cursor : o_cursor + m] = gg.offsets[1:] + v_cursor
            o_cursor += m
        v_cursor += n
    offsets[-1] = v_cursor
    return Geometry(coords, offsets)


def _filter_sig_params_global(p: dict[str, Any]) -> dict[str, Any]:
    """署名用に、ObjectRef 等の実体参照を除外したパラメータ dict を返す。"""
    try:
        from common.types import ObjectRef as _ObjectRef  # local import
    except Exception:
        _ObjectRef = object  # type: ignore
    out: dict[str, Any] = {}
    for k, v in dict(p).items():
        if isinstance(k, str) and (k.endswith("_ref") or "_ref" in k):
            continue
        if isinstance(v, _ObjectRef):
            continue
        out[k] = v
    return out


def _lazy_sig_for(lg: "LazyGeometry") -> object:
    """LazyGeometry の自己完結的な署名タプルを生成する（Core 内のみの依存）。"""
    from common.param_utils import params_signature as _params_signature  # local import

    # base
    if lg.base_kind == "geometry":
        g = lg.base_payload
        base_sig: Any = ("geom-id", id(g))
    else:
        shape_impl, params = lg.base_payload
        try:
            params_tuple = _params_signature(shape_impl, dict(params))
            base_sig = ("shape", _impl_id(shape_impl), params_tuple)
        except Exception:
            base_sig = ("shape", id(shape_impl))

    # plan
    step_sigs: list[tuple[str, tuple[tuple[str, object], ...]]] = []
    for impl, eparams in lg.plan:
        try:
            e_tuple = _params_signature(impl, _filter_sig_params_global(dict(eparams)))
        except Exception:
            e_tuple = tuple()
        step_sigs.append((_impl_id(impl), e_tuple))

    return (base_sig, tuple(step_sigs))
