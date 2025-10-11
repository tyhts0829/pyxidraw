"""
どこで: `util.cc_provider`（下位ユーティリティ層）。
何を: 現在フレームの MIDI CC スナップショット取得関数の登録/参照ポイントを提供する。
なぜ: `engine/*` から `api/*` へ依存せずに CC 値にアクセスするための依存反転ポイント。
"""

from __future__ import annotations

from typing import Callable, Mapping

# モジュールスコープでプロバイダを保持（未設定時は空の dict を返す）。
_provider: Callable[[], Mapping[int, float]] | None = None


def set_cc_snapshot_provider(provider: Callable[[], Mapping[int, float]] | None) -> None:
    """CC スナップショット取得関数を登録する。

    引数が None の場合は未設定状態に戻す。
    """

    global _provider
    _provider = provider


def get_cc_snapshot() -> Mapping[int, float]:
    """現在フレームの CC スナップショットを返す（未設定時は空）。"""

    try:
        prov = _provider
        return dict(prov()) if callable(prov) else {}
    except Exception:
        return {}


__all__ = ["set_cc_snapshot_provider", "get_cc_snapshot"]
