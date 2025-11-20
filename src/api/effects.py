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

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from common.param_utils import params_signature as _params_signature
from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.ui.parameters import get_active_runtime

from .lazy_signature import lazy_signature_for


class _PipelineCache:
    """compiled Pipeline を共有する LRU。設定やステップを含むキーで再利用を判定。"""

    def __init__(self, maxsize: int | None) -> None:
        self._cache: "OrderedDict[tuple, Pipeline]" = OrderedDict()
        self._maxsize = maxsize

    @staticmethod
    def _settings_stamp() -> tuple[float | None, int | None]:
        """量子化ステップや上限値をキーに含め、設定変更時に自然失効させる。"""
        try:
            from common.settings import get as _get_settings

            s = _get_settings()
            return (float(s.PIPELINE_QUANT_STEP), int(s.COMPILED_CACHE_MAXSIZE or -1))
        except Exception:
            return (None, None)

    def make_key(
        self,
        steps: tuple[tuple[str, tuple[tuple[str, object], ...]], ...],
        cache_maxsize: int | None,
    ) -> tuple:
        return (steps, cache_maxsize, self._settings_stamp())

    def get(self, key: tuple) -> Pipeline | None:
        try:
            p = self._cache.pop(key)
            self._cache[key] = p
            return p
        except Exception:
            return None

    def store(self, key: tuple, pipeline: Pipeline) -> None:
        if self._maxsize == 0:
            return
        self._cache[key] = pipeline
        if self._maxsize is not None and self._maxsize > 0:
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def counters(self) -> dict[str, int]:
        try:
            pipelines = list(self._cache.values())
        except Exception:
            pipelines = []
        compiled = len(pipelines)
        enabled = sum(1 for p in pipelines if getattr(p, "_cache_maxsize", 0) != 0)
        hits = sum(int(getattr(p, "_hits", 0)) for p in pipelines)
        misses = sum(int(getattr(p, "_misses", 0)) for p in pipelines)
        return {
            "compiled": compiled,
            "enabled": enabled,
            "hits": hits,
            "misses": misses,
            "step_hits": 0,
            "step_misses": 0,
        }

    def clear(self) -> None:
        self._cache.clear()


@dataclass
class Pipeline:
    steps: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    _realize_on_call: bool = False
    _cache_maxsize: int | None = 0
    _cache: "OrderedDict[tuple[object, ...], Geometry | LazyGeometry]" = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _hits: int = 0
    _misses: int = 0
    _evicts: int = 0
    # 事前コンパイル済みステップ（関数参照 + 正規化済み params）: 任意
    _compiled_steps: tuple[tuple[Callable[[Geometry], Geometry], dict[str, Any]], ...] | None = None

    def realize(self) -> "Pipeline":
        return Pipeline(
            self.steps,
            _realize_on_call=True,
            _cache_maxsize=self._cache_maxsize,
            _compiled_steps=self._compiled_steps,
        )

    def __call__(self, g: Geometry | LazyGeometry) -> Geometry | LazyGeometry:
        # 基本: LazyGeometry に plan（関数参照）を足して返す。
        # steps の事前コンパイルがあればそれを使用、無ければ都度コンパイル。
        if self._compiled_steps is None:
            from effects.registry import get_effect as _get_effect  # local import

            compiled_steps: list[tuple[Callable[[Geometry], Geometry], dict[str, Any]]] = []
            for name, params_tuple in self.steps:
                fn = _get_effect(name)
                impl = getattr(fn, "__effect_impl__", fn)
                compiled_steps.append((impl, dict(params_tuple)))
            steps_compiled = compiled_steps
        else:
            steps_compiled = list(self._compiled_steps)

        if isinstance(g, LazyGeometry):
            lg = LazyGeometry(
                base_kind=g.base_kind,
                base_payload=g.base_payload,
                plan=list(g.plan) + steps_compiled,
            )
        else:
            lg = LazyGeometry(
                base_kind="geometry",
                base_payload=g,
                plan=steps_compiled,
            )
        # キャッシュ
        key = (b"sig", lazy_signature_for(lg))
        if self._cache_maxsize != 0:  # 有効（None または 正）
            if key in self._cache:
                out_cached = self._cache.pop(key)
                self._cache[key] = out_cached
                self._hits += 1
                return out_cached
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
        # Parameter GUI 表示用のラベル（内部 UID とは分離）
        self._label_display: str | None = None

    def __getattr__(self, name: str) -> Callable[..., "PipelineBuilder"]:
        def _adder(**params: Any) -> "PipelineBuilder":
            runtime = get_active_runtime()
            resolved: dict[str, Any] = dict(params)
            # ユーザー明示の bypass は最優先（後続の GUI 値より強い）
            user_bypass_specified = False
            user_bypass_value = False
            if "bypass" in resolved:
                try:
                    user_bypass_value = bool(resolved.pop("bypass"))
                    user_bypass_specified = True
                except Exception:
                    user_bypass_value = False
                    user_bypass_specified = True
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
                resolved_map = runtime.before_effect_call(
                    step_index=len(self._steps),
                    effect_name=name,
                    fn=fn,
                    params=resolved,
                    pipeline_uid=str(self._uid or ""),
                    pipeline_label=self._label_display,
                )
                resolved = dict(resolved_map)
                impl = getattr(fn, "__effect_impl__", fn)
                # ランタイムからの bypass（GUI/永続）を吸収し、明示引数を優先
                runtime_bypass = bool(resolved.pop("bypass", False))
                bypass_final = user_bypass_value if user_bypass_specified else runtime_bypass
                if bypass_final:
                    return self  # ステップを追加せずスキップ
            else:
                impl = lambda **_: None
                # ランタイムが無い場合でも、明示 bypass があればスキップ
                if user_bypass_specified and user_bypass_value:
                    return self
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
        key = _PIPELINE_CACHE.make_key(steps_tuple, self._cache_maxsize)
        p = _PIPELINE_CACHE.get(key)
        if p is not None:
            return p
        # 事前に name→impl へ解決
        from effects.registry import get_effect as _get_effect  # local import

        compiled_steps: list[tuple[Callable[[Geometry], Geometry], dict[str, Any]]] = []
        for name, params_tuple in steps_tuple:
            fn = _get_effect(name)
            impl = getattr(fn, "__effect_impl__", fn)
            compiled_steps.append((impl, dict(params_tuple)))

        p = Pipeline(
            steps_tuple,
            _realize_on_call=False,
            _cache_maxsize=self._cache_maxsize,
            _compiled_steps=tuple(compiled_steps),
        )
        _PIPELINE_CACHE.store(key, p)
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

    # 表示ラベル（GUI のカテゴリ名）を設定
    def label(self, uid: str) -> "PipelineBuilder":
        try:
            text = str(uid).strip()
        except Exception:
            text = ""
        self._label_display = text or None
        runtime = get_active_runtime()
        if runtime is not None and self._uid is not None:
            try:
                runtime.relabel_pipeline(str(self._uid), self._label_display or "")
            except Exception:
                pass
        return self


class _EffectsAPI:
    @property
    def pipeline(self) -> PipelineBuilder:
        return PipelineBuilder()

    def label(self, uid: str) -> PipelineBuilder:
        builder = PipelineBuilder()
        return builder.label(uid)

    def __getattr__(self, name: str) -> Callable[..., PipelineBuilder]:
        def _starter(**params: Any) -> PipelineBuilder:
            builder = PipelineBuilder()
            adder = getattr(builder, name)
            return adder(**params)

        _starter.__name__ = name
        return _starter


E = _EffectsAPI()

__all__ = ["E", "Pipeline", "PipelineBuilder"]


def global_cache_counters() -> dict[str, int]:
    """Pipeline キャッシュの集計（HUD 用）。

    WeakSet はガベージ回収の影響を受けやすいため、強参照を保持する
    グローバル compiled キャッシュ（_PIPELINE_CACHE）を基準に集計する。
    """
    return _PIPELINE_CACHE.counters()


# ---- Global caches initialization (single place) ---------------------------
# 目的: 初期化順序を明確化し、重複初期化を避ける。
# COMPILED の LRU 最大数は settings 優先、失敗時は環境変数、最後に既定 128。
try:
    from common.settings import get as _get_settings

    _val = _get_settings().COMPILED_CACHE_MAXSIZE
    _PIPELINE_CACHE_MAXSIZE = int(_val) if _val is not None else 128
    if _PIPELINE_CACHE_MAXSIZE is not None and _PIPELINE_CACHE_MAXSIZE < 0:
        _PIPELINE_CACHE_MAXSIZE = 0
except Exception:
    try:
        import os as _os

        _raw = _os.getenv("PXD_COMPILED_CACHE_MAXSIZE")
        _PIPELINE_CACHE_MAXSIZE = int(_raw) if _raw is not None else 128
        if _PIPELINE_CACHE_MAXSIZE is not None and _PIPELINE_CACHE_MAXSIZE < 0:
            _PIPELINE_CACHE_MAXSIZE = 0
    except Exception:
        _PIPELINE_CACHE_MAXSIZE = 128

# 単一インスタンスの LRU
_PIPELINE_CACHE = _PipelineCache(_PIPELINE_CACHE_MAXSIZE)


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
