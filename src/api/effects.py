"""
どこで: `api.effects`（エフェクト・パイプラインの高レベル API）。
何を: 登録エフェクトの直列適用を“宣言”し、`LazyGeometry` に plan を積むビルダー/パイプラインを提供。
なぜ: effect を既定で遅延化し、終端で一括 `realize()` するため。

提供コンポーネント:
- `Pipeline`: 宣言（ステップ列）を保持し、`__call__(g)` で `LazyGeometry` に plan を連結する。
- `PipelineBuilder`: チェーン可能な薄いビルダー（`.build()` で `Pipeline` を返す）。
- `E`: 利用者向けシングルトン（`from api import E`）。`E.pipeline` が `PipelineBuilder` を返す。
"""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from common.param_utils import params_signature as _params_signature
from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.ui.parameters import get_active_runtime

from .lazy_signature import lazy_signature_for

_GLOBAL_PIPELINES: "weakref.WeakSet[Pipeline]"  # forward decl
_GLOBAL_COMPILED: "OrderedDict[tuple, Pipeline]"  # forward compiled cache
_GLOBAL_COMPILED_MAXSIZE_ENV = os.getenv("PXD_COMPILED_CACHE_MAXSIZE")
try:
    _GLOBAL_COMPILED_MAXSIZE: int | None = (
        int(_GLOBAL_COMPILED_MAXSIZE_ENV) if _GLOBAL_COMPILED_MAXSIZE_ENV is not None else 128
    )
    if _GLOBAL_COMPILED_MAXSIZE is not None and _GLOBAL_COMPILED_MAXSIZE < 0:
        _GLOBAL_COMPILED_MAXSIZE = 0
except Exception:
    _GLOBAL_COMPILED_MAXSIZE = 128


@dataclass
class Pipeline:
    steps: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    _realize_on_call: bool = False
    _cache_maxsize: int | None = 0
    _cache: "OrderedDict[tuple[object, ...], Geometry | LazyGeometry]" = dataclass(init=False, repr=False)  # type: ignore[assignment]
    _hits: int = 0
    _misses: int = 0
    _evicts: int = 0

    def __post_init__(self) -> None:  # noqa: D401
        self._cache = OrderedDict()
        try:
            _GLOBAL_PIPELINES.add(self)
        except Exception:
            pass

    def realize(self) -> "Pipeline":
        return Pipeline(self.steps, _realize_on_call=True)

    def __call__(self, g: Geometry | LazyGeometry) -> Geometry | LazyGeometry:
        # 基本: LazyGeometry に plan（関数参照）を足して返す。
        # ここで effects.registry を解決し、engine.core 側には関数参照のみを渡す。
        from effects.registry import get_effect as _get_effect  # local import

        compiled_steps: list[tuple[Callable[[Geometry], Geometry], dict[str, Any]]] = []
        for name, params_tuple in self.steps:
            fn = _get_effect(name)
            impl = getattr(fn, "__effect_impl__", fn)
            compiled_steps.append((impl, dict(params_tuple)))

        if isinstance(g, LazyGeometry):
            lg = LazyGeometry(
                base_kind=g.base_kind,
                base_payload=g.base_payload,
                plan=list(g.plan) + compiled_steps,
            )
        else:
            lg = LazyGeometry(
                base_kind="geometry",
                base_payload=g,
                plan=compiled_steps,
            )
        # キャッシュ
        key = (b"sig", lazy_signature_for(lg))
        if self._cache_maxsize != 0:  # 有効（None または 正）
            if key in self._cache:
                out = self._cache.pop(key)
                self._cache[key] = out
                self._hits += 1
                return out
        out: Geometry | LazyGeometry = lg.realize() if self._realize_on_call else lg
        if self._cache_maxsize != 0:
            self._cache[key] = out
            if self._cache_maxsize is not None and self._cache_maxsize > 0:
                while len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)
                    self._evicts += 1
            self._misses += 1
        return out

    # ---- cache helpers ----
    # 旧: _cache_key/_tuple_from_plan は署名ベースに移行したため削除

    def clear_cache(self) -> None:
        self._cache.clear()


class PipelineBuilder:
    def __init__(self) -> None:
        self._steps: list[tuple[str, tuple[tuple[str, object], ...]]] = []
        self._cache_maxsize: int | None = None
        self._uid: str | None = None

    def __getattr__(self, name: str) -> Callable[..., "PipelineBuilder"]:
        def _adder(**params: Any) -> "PipelineBuilder":
            runtime = get_active_runtime()
            resolved = dict(params)
            if runtime is not None:
                # 初回に Pipeline UID を確保
                if self._uid is None:
                    try:
                        self._uid = runtime.next_pipeline_uid()
                    except Exception:
                        self._uid = ""
                # 実関数（署名/メタ用）は effects.registry から取得
                from effects.registry import get_effect as _get_effect  # local import

                fn = _get_effect(name)
                resolved = runtime.before_effect_call(
                    step_index=len(self._steps),
                    effect_name=name,
                    fn=fn,
                    params=resolved,
                    pipeline_uid=str(self._uid or ""),
                )
                impl = getattr(fn, "__effect_impl__", fn)
            else:
                impl = lambda **_: None
            params_tuple = _params_signature(impl, dict(resolved))
            self._steps.append((name, params_tuple))
            return self

        _adder.__name__ = name
        return _adder

    # 互換: すぐに適用できるよう callable を返す
    def __call__(self, g: Geometry | LazyGeometry) -> Geometry | LazyGeometry:
        return self.build()(g)

    def build(self) -> Pipeline:
        steps_tuple = tuple(self._steps)
        key = (steps_tuple, self._cache_maxsize)
        # グローバル compiled を再利用
        try:
            p = _GLOBAL_COMPILED.get(key)  # type: ignore[arg-type]
        except Exception:
            p = None
        if p is not None:
            return p
        p = Pipeline(steps_tuple, _realize_on_call=False, _cache_maxsize=self._cache_maxsize)
        try:
            _GLOBAL_COMPILED[key] = p  # type: ignore[index]
            if _GLOBAL_COMPILED_MAXSIZE is not None and _GLOBAL_COMPILED_MAXSIZE > 0:
                while len(_GLOBAL_COMPILED) > _GLOBAL_COMPILED_MAXSIZE:
                    _GLOBAL_COMPILED.popitem(last=False)
        except Exception:
            pass
        return p

    # 明示バリア（Geometry にする Pipeline を返す）
    def realize(self) -> Pipeline:
        return Pipeline(
            tuple(self._steps), _realize_on_call=True, _cache_maxsize=self._cache_maxsize
        )

    # キャッシュ設定
    def cache(self, *, maxsize: int | None) -> "PipelineBuilder":
        self._cache_maxsize = maxsize
        return self


class _EffectsAPI:
    @property
    def pipeline(self) -> PipelineBuilder:
        return PipelineBuilder()


E = _EffectsAPI()

__all__ = ["E", "Pipeline", "PipelineBuilder"]


def global_cache_counters() -> dict[str, int]:
    """Pipeline キャッシュの集計（HUD 用）。

    WeakSet はガベージ回収の影響を受けやすいため、強参照を保持する
    グローバル compiled キャッシュ（_GLOBAL_COMPILED）を基準に集計する。
    """
    try:
        pipelines = list(_GLOBAL_COMPILED.values())
    except Exception:
        pipelines = []
    compiled = len(pipelines)
    enabled = sum(1 for p in pipelines if getattr(p, "_cache_maxsize", 0) != 0)
    hits = sum(int(getattr(p, "_hits", 0)) for p in pipelines)
    misses = sum(int(getattr(p, "_misses", 0)) for p in pipelines)
    step_hits = 0
    step_misses = 0
    return {
        "compiled": compiled,
        "enabled": enabled,
        "hits": hits,
        "misses": misses,
        "step_hits": step_hits,
        "step_misses": step_misses,
    }


try:
    _GLOBAL_PIPELINES = weakref.WeakSet()
except Exception:  # pragma: no cover
    _GLOBAL_PIPELINES = weakref.WeakSet()  # type: ignore[assignment]
try:
    _GLOBAL_COMPILED = OrderedDict()
except Exception:  # pragma: no cover
    _GLOBAL_COMPILED = OrderedDict()  # type: ignore[assignment]


def _is_json_like(obj: object) -> bool:
    """最小の JSON 風判定（テスト用の互換ユーティリティ）。"""
    if obj is None:
        return True
    if isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_is_json_like(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, (str, int)) and _is_json_like(v) for k, v in obj.items())
    return False
