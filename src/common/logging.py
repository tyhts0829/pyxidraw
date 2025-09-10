"""
プロジェクト向けの軽量ロギングユーティリティ（提案 7）。

要点:
- 既定では各モジュールが `logging.getLogger(__name__)` でロガーを取得する。
- アプリ側で設定が無い場合でも、妥当な最小構成を 1 度だけ適用するヘルパーを提供する。
"""

from __future__ import annotations

import logging


def setup_default_logging(level: int | str = "INFO") -> None:
    """最小限のロギング設定を 1 度だけ適用する。

    - ルートロガーにハンドラが既にあれば何もしない（no-op）
    - 上位のランナー/CLI から呼び出す想定
    """
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)

    root = logging.getLogger()
    if root.handlers:
        # Assume the app has configured logging
        return
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


__all__ = ["setup_default_logging"]
