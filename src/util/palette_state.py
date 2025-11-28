from __future__ import annotations

"""パレット状態の共有ユーティリティ。

どこで: `util` 層。
何を: 現在有効なパレットオブジェクトを保持するための最小限の set/get API を提供する。
なぜ: engine 層と api 層の間で依存方向を崩さずにパレットを共有するため。

設計メモ:
- 本モジュールはパレット型に依存しない（`object` として保持する）。
- 実際の型解釈は呼び出し側（例: `api.palette` や `engine.ui.palette`）が担う。
"""

from typing import Any

_PALETTE_OBJ: Any | None = None


def set_palette(obj: Any | None) -> None:
    """現在のパレットオブジェクトを設定する。"""
    global _PALETTE_OBJ
    _PALETTE_OBJ = obj


def get_palette() -> Any | None:
    """現在のパレットオブジェクトを返す（未設定時は None）。"""
    return _PALETTE_OBJ


__all__ = ["set_palette", "get_palette"]
