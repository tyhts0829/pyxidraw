"""
どこで: `api.lazy_signature`
何を: LazyGeometry（shape spec + effect chain）から決定的な署名（blake2b-128）を生成。
なぜ: realize 以前から安定キーでキャッシュ/共有を行うため。
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

from common.func_id import impl_id as _impl_id
from common.param_utils import params_signature as _params_signature
from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry


def _digest_for_params_tuple(params_tuple: tuple[tuple[str, object], ...]) -> bytes:
    data = repr(params_tuple).encode()
    return hashlib.blake2b(data, digest_size=8).digest()


def _freeze_plan(
    plan: Sequence[tuple[Callable[[Geometry], Geometry], Mapping[str, Any]]],
) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
    items: list[tuple[str, tuple[tuple[str, object], ...]]] = []
    for impl, params in plan:
        params_tuple = _params_signature(impl, dict(params))
        items.append((_impl_id(impl), params_tuple))
    return tuple(items)


def lazy_signature_for(lg: LazyGeometry) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    # base
    if lg.base_kind == "geometry":
        g = lg.base_payload
        assert isinstance(g, Geometry)
        h.update(b"geom-id")
        h.update(str(id(g)).encode())
    else:
        # base_payload: (shape_name, shape_impl, params_dict)
        try:
            _shape_name, shape_impl, params = lg.base_payload
        except Exception:
            shape_impl, params = lg.base_payload  # 後方互換
        params_tuple = _params_signature(shape_impl, dict(params))
        h.update(b"shape")
        h.update(_impl_id(shape_impl).encode())
        h.update(_digest_for_params_tuple(params_tuple))
    # plan
    h.update(b"plan")
    for impl_id, params_tuple in _freeze_plan(lg.plan):
        h.update(impl_id.encode())
        h.update(_digest_for_params_tuple(params_tuple))
    dig = h.digest()
    return dig
