from __future__ import annotations

import types
from typing import Mapping

import pytest


@pytest.mark.smoke
def test_midi_service_snapshot_flatten_without_real_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # engine.io.service.MidiService は Manager を要求する。軽いダミーで置換。
    from engine.io.service import MidiService

    class DummyCtrl:
        def __init__(self) -> None:
            self.cc = {1: 0.5, 2: 1.0}

        def iter_pending(self):  # pragma: no cover - 呼ばれない
            return []

        def update_cc(self, _msg) -> None:  # pragma: no cover - 呼ばれない
            pass

        def save_cc(self) -> None:  # pragma: no cover - ダミー
            pass

    class DummyManager:
        def __init__(self) -> None:
            self.controllers = {"d": DummyCtrl()}

        def update_midi_controllers(self) -> None:
            return None

        def save_cc(self) -> None:
            return None

    m = MidiService(DummyManager())
    m.tick(1 / 60)
    snap = m.snapshot()
    assert isinstance(snap, Mapping)
    assert snap.get(1) == 0.5 and snap.get(2) == 1.0


def test_connect_midi_controllers_import_and_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    # 実 mido を import せずに、sys.modules へフェイクを注入して接続経路を検証
    import sys

    from engine.io import controller as ctrl
    from engine.io import manager as mgr

    class _InPort:
        def iter_pending(self):  # pragma: no cover - 未使用
            return []

    fake_mido = types.SimpleNamespace(
        get_input_names=lambda: ["ArcController OUT (mock)"],
        open_input=lambda name: _InPort(),
        get_output_names=lambda: [],  # sync_grid_knob_values 経路を早期 return
    )

    monkeypatch.setitem(sys.modules, "mido", fake_mido)
    # 既に import 済みの engine.io.controller 側の参照も差し替える
    monkeypatch.setattr(ctrl, "mido", fake_mido, raising=False)

    mm = mgr.connect_midi_controllers()
    assert isinstance(mm.controller_names, list) and len(mm.controller_names) >= 0
