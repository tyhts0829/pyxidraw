"""
どこで: `common.func_id`
何を: 関数オブジェクトから安定 ID を取得するユーティリティ（`module:qualname`）。
なぜ: 署名/キャッシュ鍵の一貫性を保ち、重複実装を排除するため。
"""

from __future__ import annotations

from typing import Any


def impl_id(fn: Any) -> str:
    """関数の一意 ID を返す（`module:qualname` を基本とし、失敗時は `id()` を用いる）。

    Parameters
    ----------
    fn : Any
        対象の関数または呼び出し可能。

    Returns
    -------
    str
        `"{module}:{qualname}"` 形式の ID（取得失敗時は `str(id(fn))`）。
    """
    try:
        mod = getattr(fn, "__module__", "") or ""
        qn = getattr(fn, "__qualname__", getattr(fn, "__name__", "")) or ""
        s = f"{mod}:{qn}".strip(":")
        return s or str(id(fn))
    except Exception:
        return str(id(fn))
