"""
どこで: `api.lfo`
何を: LFO（低周波オシレータ）のファクトリ関数を公開する薄いファサード。
なぜ: 利用者が `from api import lfo` で取得できるようにするため（実装は `common.lfo`）。
"""

from __future__ import annotations

from common.lfo import LFO, lfo

__all__ = ["lfo", "LFO"]
