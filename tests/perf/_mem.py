"""
テスト専用: メモリ計測ユーティリティ。

psutil の RSS 差分と tracemalloc のピーク割当を取得する簡易関数を提供する。
ベンチマーク自体の指標は pytest-benchmark に委ね、ここでは補助的なメモリ情報のみ扱う。
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Tuple

try:  # psutil が無い環境でもテストを壊さない
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback
    psutil = None  # type: ignore
import tracemalloc


def measure_memory(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, dict]:
    """関数実行時の RSS 差分と tracemalloc ピークを測定して返す。

    返り値は `(result, {"rss_delta": int, "alloc_peak": int})`。
    """
    proc = psutil.Process() if psutil is not None else None  # type: ignore
    gc.collect()
    rss_before = proc.memory_info().rss if proc is not None else 0  # type: ignore[union-attr]
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_after = proc.memory_info().rss if proc is not None else 0  # type: ignore[union-attr]
    return result, {"rss_delta": int(rss_after - rss_before), "alloc_peak": int(peak)}
