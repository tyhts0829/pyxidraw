from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from engine.core.geometry import Geometry
import effects  # 副作用で標準エフェクトを登録
from effects.registry import get_effect


def _geometry_hash(g: Geometry) -> bytes:
    c, o = g.as_arrays(copy=False)
    c = np.ascontiguousarray(c).view(np.uint8)
    o = np.ascontiguousarray(o).view(np.uint8)
    h = hashlib.blake2b(digest_size=16)
    h.update(c)
    h.update(o)
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
    def __init__(self, steps: Sequence[Step]):
        self._steps = list(steps)
        self._cache: Dict[Tuple[bytes, bytes], Geometry] = {}

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
            return self._cache[key]

        out = g
        for st in self._steps:
            fn = get_effect(st.name)
            out = fn(out, **st.params)

        # 単層キャッシュ
        self._cache[key] = out
        return out

    # ---- Serialization (Proposal 6) ----
    def to_spec(self) -> List[Dict[str, Any]]:
        """Return a serializable spec: [{"name": str, "params": dict}]."""
        return [{"name": s.name, "params": dict(s.params)} for s in self._steps]

    @staticmethod
    def from_spec(spec: Sequence[Dict[str, Any]]) -> "Pipeline":
        """Create a Pipeline from a spec. Raises on invalid shape or effect name."""
        if not isinstance(spec, (list, tuple)):
            raise TypeError("spec must be a list/tuple of steps")
        steps: List[Step] = []
        for i, entry in enumerate(spec):
            if not isinstance(entry, dict):
                raise TypeError(f"spec[{i}] must be a dict, got {type(entry).__name__}")
            name = entry.get("name")
            params = entry.get("params", {})
            if not isinstance(name, str):
                raise TypeError(f"spec[{i}]['name'] must be str")
            if not isinstance(params, dict):
                raise TypeError(f"spec[{i}]['params'] must be dict")
            # Validate registration early
            get_effect(name)
            steps.append(Step(name, params))
        return Pipeline(steps)


class PipelineBuilder:
    def __init__(self):
        self._steps: List[Step] = []

    def _add(self, name: str, params: Dict[str, Any]) -> "PipelineBuilder":
        self._steps.append(Step(name, params))
        return self

    def __getattr__(self, name: str):
        # 動的にエフェクト名を受け取り、paramsを蓄積
        def adder(**params):
            return self._add(name, params)

        return adder

    def build(self) -> Pipeline:
        return Pipeline(self._steps)

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
