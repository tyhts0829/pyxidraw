"""
どこで: `api.sketch_runner.midi`
何を: MIDI 初期化（Null 実装含む）とスナップショット関数の提供。
なぜ: `api.sketch` から初期化責務を分離し、フォールバックを簡潔に保つため。
"""

from __future__ import annotations

import logging
from typing import Mapping

logger = logging.getLogger(__name__)


def setup_midi(_use_midi: bool):
    """MIDI を初期化して (manager, service, snapshot_fn) を返す。

    - `_use_midi` が False の場合は Null 実装を返す。
    - 例外/未接続時も安全に Null へフォールバックする。
    """

    class _NullMidi:
        def snapshot(self) -> Mapping[int, float]:
            return {}

        def tick(self, dt: float) -> None:  # noqa: ARG002
            return None

    if not _use_midi:
        nm = None
        ns = _NullMidi()
        return nm, ns, ns.snapshot

    try:
        # 遅延インポート（依存未導入環境でもフォールバック可能に）
        from engine.io.manager import connect_midi_controllers
        from engine.io.service import MidiService

        midi_manager_local = connect_midi_controllers()
        # 0台接続もエラー扱いにするかは strict で切替（現状はエラー扱い）
        if not getattr(midi_manager_local, "controllers", {}):
            raise RuntimeError("MIDI デバイスが接続されていません")
        midi_service_local = MidiService(midi_manager_local)
        return midi_manager_local, midi_service_local, midi_service_local.snapshot
    except Exception as e:  # ImportError / InvalidPortError / RuntimeError など
        logger.warning("MIDI unavailable; falling back to NullMidi: %s", e)
        nm = None
        ns = _NullMidi()
        return nm, ns, ns.snapshot


__all__ = ["setup_midi"]
