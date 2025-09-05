from __future__ import annotations

import hashlib
from dataclasses import dataclass
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence, Tuple
import inspect

import numpy as np

from engine.core.geometry import Geometry
import effects  # 副作用で標準エフェクトを登録
from effects.registry import get_effect


def _geometry_hash(g: Geometry) -> bytes:
    """Geometry のハッシュ。

    Geometry が `digest` プロパティを提供する場合はそれを優先し、
    なければ従来通りに coords/offsets から計算します。
    """
    try:
        return g.digest  # type: ignore[attr-defined]
    except Exception:
        # フォールバック（理論上到達しない想定）
        c, o = g.as_arrays(copy=False)
        c = np.ascontiguousarray(c).view(np.uint8)
        o = np.ascontiguousarray(o).view(np.uint8)
        h = hashlib.blake2b(digest_size=16)
        # Fallback 経路のためコピーを許容し、bytes に変換して渡す
        h.update(c.tobytes())
        h.update(o.tobytes())
        return h.digest()


def _fn_version(fn: Callable[..., Geometry]) -> bytes:
    code = getattr(fn, "__code__", None)
    data = code.co_code if code else repr(fn).encode()
    return hashlib.blake2b(data, digest_size=8).digest()


def _params_digest(params: Dict[str, Any]) -> bytes:
    # 正規化して決定的にシリアライズ
    def make_hashable(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        if isinstance(obj, (list, tuple)):
            return tuple(make_hashable(x) for x in obj)
        if isinstance(obj, np.ndarray):
            return tuple(obj.flatten().tolist())
        return obj

    normalized = make_hashable(params)
    data = repr(normalized).encode()
    return hashlib.blake2b(data, digest_size=8).digest()


@dataclass(frozen=True)
class Step:
    name: str
    params: Dict[str, Any]


class Pipeline:
    def __init__(self, steps: Sequence[Step], *, cache_maxsize: int | None = None):
        self._steps = list(steps)
        # LRU 互換の単層キャッシュ（maxsize=None なら従来通り無制限）
        self._cache_maxsize: int | None = cache_maxsize
        self._cache: "OrderedDict[Tuple[bytes, bytes], Geometry]" = OrderedDict()

        # パイプラインハッシュを先に計算
        h = hashlib.blake2b(digest_size=16)
        for st in self._steps:
            fn = get_effect(st.name)
            h.update(st.name.encode())
            h.update(_fn_version(fn))
            h.update(_params_digest(st.params))
        self._pipeline_key = h.digest()

    def __call__(self, g: Geometry) -> Geometry:
        key = (_geometry_hash(g), self._pipeline_key)
        if key in self._cache:
            # LRU: 参照を末尾へ
            out = self._cache.pop(key)
            self._cache[key] = out
            return out

        out = g
        for st in self._steps:
            fn = get_effect(st.name)
            out = fn(out, **st.params)

        # 単層キャッシュ（LRU 風）
        if self._cache_maxsize == 0:
            return out  # キャッシュ無効
        self._cache[key] = out
        if self._cache_maxsize is not None and self._cache_maxsize > 0:
            while len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)  # 先頭（最古）を追い出す
        return out

    def clear_cache(self) -> None:
        """パイプラインの単層キャッシュをクリア。"""
        self._cache.clear()

    def __repr__(self) -> str:  # 開発時の可読性向上
        steps = ", ".join(f"{s.name}({', '.join(f'{k}={v!r}' for k, v in s.params.items())})" for s in self._steps)
        return f"Pipeline(steps=[{steps}], cache_maxsize={self._cache_maxsize})"

    __str__ = __repr__

    # ---- Serialization (Proposal 6) ----
    def to_spec(self) -> List[Dict[str, Any]]:
        """Return a serializable spec: [{"name": str, "params": dict}]."""
        return [{"name": s.name, "params": dict(s.params)} for s in self._steps]

    @staticmethod
    def from_spec(spec: Sequence[Dict[str, Any]]) -> "Pipeline":
        """Create a Pipeline from a spec. Raises on invalid shape or effect name."""
        validate_spec(spec)
        steps: List[Step] = [Step(str(entry["name"]), dict(entry.get("params", {}))) for entry in spec]  # type: ignore[arg-type]
        return Pipeline(steps)


class PipelineBuilder:
    def __init__(self):
        self._steps: List[Step] = []
        # 既定サイズは環境変数から上書き可能
        self._cache_maxsize: int | None = None
        self._strict: bool = False
        _env = os.getenv("PXD_PIPELINE_CACHE_MAXSIZE")
        if _env is not None:
            try:
                val = int(_env)
                self._cache_maxsize = val
            except ValueError:
                pass

    def _add(self, name: str, params: Dict[str, Any]) -> "PipelineBuilder":
        self._steps.append(Step(name, params))
        return self

    def __getattr__(self, name: str):
        # 動的にエフェクト名を受け取り、paramsを蓄積
        def adder(**params):
            return self._add(name, params)

        return adder

    # オプション: 単層キャッシュの上限を設定（None で無制限・従来互換）
    def cache(self, *, maxsize: int | None) -> "PipelineBuilder":
        self._cache_maxsize = maxsize
        return self

    # オプション: 厳格検証を有効化（ビルド時にパラメータ名を検査）
    def strict(self, enabled: bool = True) -> "PipelineBuilder":
        """Enable strict parameter-name validation at build time.

        - 各ステップのパラメータ名をエフェクト関数のシグネチャと突き合わせ、未知キーがあれば TypeError を送出。
        - `**kwargs` を受け取る関数は除外（未知キー許容）。
        """
        self._strict = enabled
        return self

    def build(self) -> Pipeline:
        if self._strict:
            for i, st in enumerate(self._steps):
                fn = get_effect(st.name)
                try:
                    sig = inspect.signature(fn)
                except ValueError:
                    # Builtins などはスキップ
                    continue
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if has_var_kw:
                    continue
                allowed = {
                    p.name
                    for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                }
                if "g" in allowed:
                    allowed.remove("g")
                unknown = [k for k in st.params.keys() if k not in allowed]
                if unknown:
                    allowed_sorted = ", ".join(sorted(allowed))
                    raise TypeError(
                        f"step[{i}] effect '{st.name}' has unknown params: {unknown}. Allowed: [{allowed_sorted}]"
                    )
        return Pipeline(self._steps, cache_maxsize=self._cache_maxsize)

    def __call__(self, g: Geometry) -> Geometry:
        return self.build()(g)


class Effects:
    @property
    def pipeline(self) -> PipelineBuilder:
        return PipelineBuilder()


# シングルトンインスタンス（従来の `from api import E` に対応）
E = Effects()

# Helper functions (optional API)
def to_spec(pipeline: Pipeline) -> List[Dict[str, Any]]:
    return pipeline.to_spec()


def from_spec(spec: Sequence[Dict[str, Any]]) -> Pipeline:
    return Pipeline.from_spec(spec)


# ---- Spec validation (Proposal 6) -----------------------------------------
def _is_json_like(value: Any) -> bool:
    """Heuristic check whether a value is JSON-like (serializable)."""
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_like(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_like(v) for k, v in value.items())
    return False


def validate_spec(spec: Sequence[Dict[str, Any]]) -> None:
    """Validate a pipeline spec. Raises TypeError/KeyError on failure.

    Rules:
    - spec is a list/tuple of {"name": str, "params": dict}
    - effect name must be registered
    - params must be dict and JSON-like (numbers/strings/bools/None, nested lists/dicts)
    - parameter names are checked against function signature when possible
      (unknown keys are allowed if the function accepts **kwargs)
    """
    if not isinstance(spec, (list, tuple)):
        raise TypeError("spec must be a list or tuple of steps")

    for i, entry in enumerate(spec):
        if not isinstance(entry, dict):
            raise TypeError(f"spec[{i}] must be a dict, got {type(entry).__name__}")
        name = entry.get("name")
        params = entry.get("params", {})
        if not isinstance(name, str):
            raise TypeError(f"spec[{i}]['name'] must be str")
        if not isinstance(params, dict):
            raise TypeError(f"spec[{i}]['params'] must be dict")

        # Validate effect registration
        fn = get_effect(name)  # raises KeyError if not registered

        # Validate params JSON-likeness
        for k, v in params.items():
            if not isinstance(k, str):
                raise TypeError(f"spec[{i}]['params'] key must be str: {k!r}")
            if not _is_json_like(v):
                raise TypeError(f"spec[{i}]['params']['{k}'] is not JSON-serializable-like: {type(v).__name__}")

        # Validate parameter names against signature if possible
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kw:
                # allowed keywords are those after the first positional 'g' argument
                allowed = {p.name for p in sig.parameters.values() if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)}
                if 'g' in allowed:
                    allowed.remove('g')
                unknown = [k for k in params.keys() if k not in allowed]
                if unknown:
                    allowed_sorted = sorted(allowed)
                    raise TypeError(
                        "spec[{}] has unknown params for effect '{}': {}. Allowed: {}".format(
                            i, name, unknown, allowed_sorted
                        )
                    )
        except ValueError:
            # Builtins or objects without signature: skip strict check
            pass

        # Optional: param meta validation (type/range/choices) if effect exposes __param_meta__
        meta = getattr(fn, "__param_meta__", None)
        if isinstance(meta, dict):
            for k, rules in meta.items():
                if k not in params:
                    continue
                v = params[k]
                # type check (loose)
                t = rules.get("type") if isinstance(rules, dict) else None
                if t == "number" and not isinstance(v, (int, float)):
                    raise TypeError(f"spec[{i}]['params']['{k}'] must be number, got {type(v).__name__}")
                if t == "integer" and not isinstance(v, int):
                    raise TypeError(f"spec[{i}]['params']['{k}'] must be integer, got {type(v).__name__}")
                if t == "string" and not isinstance(v, str):
                    raise TypeError(f"spec[{i}]['params']['{k}'] must be string, got {type(v).__name__}")
                if t == "vec3":
                    # allow scalar, 1-tuple, or 3-tuple of numbers
                    def _is_num(x):
                        return isinstance(x, (int, float))
                    if _is_num(v):
                        pass
                    elif isinstance(v, (list, tuple)) and len(v) in (1, 3) and all(_is_num(x) for x in v):
                        pass
                    else:
                        raise TypeError(f"spec[{i}]['params']['{k}'] must be scalar, 1-tuple, or 3-tuple of numbers")
                # range
                if isinstance(rules, dict):
                    if "min" in rules and isinstance(v, (int, float)) and v < rules["min"]:
                        raise TypeError(f"spec[{i}]['params']['{k}']={v} is below min {rules['min']}")
                    if "max" in rules and isinstance(v, (int, float)) and v > rules["max"]:
                        raise TypeError(f"spec[{i}]['params']['{k}']={v} exceeds max {rules['max']}")
                    # choices
                    choices = rules.get("choices")
                    if choices is not None and v not in choices:
                        raise TypeError(f"spec[{i}]['params']['{k}']={v!r} must be one of {choices}")
