"""
どこで: `common.env`
何を: 環境変数の軽量パースヘルパを提供。
なぜ: 各所に散在する `os.getenv` + 例外/境界ガードを簡素化するため。
"""

from __future__ import annotations

import os
from typing import Optional


def env_int(
    name: str, default: Optional[int] = None, *, min_value: Optional[int] = None
) -> Optional[int]:
    """整数環境変数を取得（存在しない/不正値は既定値）。

    Parameters
    ----------
    name : str
        環境変数名。
    default : Optional[int]
        既定値（`None` を渡すと `None` を許容）。
    min_value : Optional[int]
        下限（指定時、結果が下回れば下限に丸める）。

    Returns
    -------
    Optional[int]
        取得した整数値。未設定/不正時は `default` を返す。
    """
    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        val = int(raw)
        if min_value is not None and val < min_value:
            val = min_value
        return val
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    """真偽環境変数を取得（0/1, true/false を許容）。"""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    try:
        # 数値優先
        return int(raw) != 0
    except Exception:
        s = raw.strip().lower()
        if s in {"true", "t", "yes", "y", "on"}:
            return True
        if s in {"false", "f", "no", "n", "off"}:
            return False
        return bool(default)
