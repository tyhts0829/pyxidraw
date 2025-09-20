"""
どこで: `engine.ui.parameters` のメタ解決層。
何を: 関数の docstring/シグネチャ/`__param_meta__` を解決してキャッシュし、`FunctionInfo` として提供。
なぜ: ランタイム解決コストの低減と、値解決の決定性/再現性を高めるため。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FunctionInfo:
    """関数の docstring・シグネチャ・パラメータメタをまとめた情報。"""

    kind: str
    name: str
    doc: str | None
    signature: inspect.Signature | None
    param_meta: dict[str, Mapping[str, Any]]


class FunctionIntrospector:
    """関数のドキュメント/シグネチャ/メタ情報をキャッシュ付きで提供する。"""

    def __init__(self) -> None:
        self._doc_cache: dict[str, str | None] = {}
        self._signature_cache: dict[str, inspect.Signature | None] = {}
        self._meta_cache: dict[str, dict[str, Mapping[str, Any]]] = {}

    def resolve(self, *, kind: str, name: str, fn: Any) -> FunctionInfo:
        """指定された関数のメタ情報を取得する。"""
        cache_key = f"{kind}::{name}"
        doc = self._resolve_doc(cache_key, fn)
        signature = self._resolve_signature(cache_key, fn)
        param_meta = self._resolve_meta(cache_key, fn)
        return FunctionInfo(
            kind=kind, name=name, doc=doc, signature=signature, param_meta=param_meta
        )

    def _resolve_doc(self, cache_key: str, fn: Any) -> str | None:
        if cache_key not in self._doc_cache:
            doc = inspect.getdoc(fn)
            self._doc_cache[cache_key] = doc.splitlines()[0] if doc else None
        return self._doc_cache[cache_key]

    def _resolve_signature(self, cache_key: str, fn: Any) -> inspect.Signature | None:
        if cache_key not in self._signature_cache:
            try:
                signature = inspect.signature(fn)
            except (TypeError, ValueError):
                signature = None
            self._signature_cache[cache_key] = signature
        return self._signature_cache[cache_key]

    def _resolve_meta(self, cache_key: str, fn: Any) -> dict[str, Mapping[str, Any]]:
        if cache_key not in self._meta_cache:
            meta_raw = getattr(fn, "__param_meta__", None)
            if isinstance(meta_raw, Mapping):
                normalized: dict[str, Mapping[str, Any]] = {}
                for param_name, value in meta_raw.items():
                    if isinstance(value, Mapping):
                        normalized[str(param_name)] = dict(value)
                self._meta_cache[cache_key] = normalized
            else:
                self._meta_cache[cache_key] = {}
        return self._meta_cache[cache_key]


__all__ = ["FunctionInfo", "FunctionIntrospector"]
