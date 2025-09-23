"""
どこで: engine.io
何を: CC スナップショットの総域（total）マッピングを提供。
なぜ: draw(t, cc) 内で `cc[idx]` を安全に参照できるようにし、初回フレームでも KeyError を避けるため。
"""

from __future__ import annotations

from collections.abc import Mapping, KeysView, ItemsView, Iterator
from typing import Any, TypeVar, overload


class CCSnapshot(Mapping[int, float]):
    """未登録キーに対しても 0.0 を返す CC マッピング。

    - `cc[idx]` は常に `float`（未登録は 0.0）。
    - `.keys()`/`.items()` は観測済みのキーのみ列挙。
    - `repr(cc)` は `CCSnapshot({...}, default=0.0)` の形式。
    - `raw()` で観測済み辞書を取得できる。
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[int, float] | None = None) -> None:
        self._data: dict[int, float] = dict(data or {})

    # Mapping interface
    def __getitem__(self, key: int) -> float:  # type: ignore[override]
        try:
            return float(self._data[int(key)])
        except Exception:
            return 0.0

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        return iter(self._data.keys())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._data)

    # convenience
    _T = TypeVar("_T")

    @overload
    def get(self, key: int) -> float | None:  # pragma: no cover - typing overload
        ...

    @overload
    def get(self, key: int, default: _T) -> float | _T:  # pragma: no cover - typing overload
        ...

    def get(self, key: int, default: Any = None) -> Any:
        try:
            return float(self._data[int(key)])
        except Exception:
            # 既定は 0.0（Mapping 契約上は Optional だが実運用は 0.0 を推奨）
            return 0.0 if default is None else default

    def keys(self) -> KeysView[int]:  # type: ignore[override]
        return self._data.keys()

    def items(self) -> ItemsView[int, float]:  # type: ignore[override]
        return self._data.items()

    def raw(self) -> dict[int, float]:
        """観測済みのスナップショット（浅いコピー）を返す。"""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"CCSnapshot({self._data!r}, default=0.0)"

    # 拡張: from_dict ヘルパ（任意）
    @classmethod
    def from_dict(cls, d: Mapping[int, Any]) -> "CCSnapshot":
        try:
            return cls({int(k): float(v) for k, v in d.items()})
        except Exception:
            return cls(dict(d))


__all__ = ["CCSnapshot"]
