import pytest

# What this tests (TEST_HARDENING_PLAN.md §Optional)
# - Without opening real MIDI ports, MidiController can process a 7-bit CC message
#   and static helpers work; ensures mido integration imports and basic logic are healthy.

mido = pytest.importorskip("mido")

from engine.io.controller import MidiController


@pytest.mark.optional
def test_mido_message_processed_without_device_connection():
    # construct a controller instance without running __init__ (avoids opening ports)
    mc = MidiController.__new__(MidiController)
    mc.mode = "7bit"

    # 7-bit CC message should be handled
    msg = mido.Message("control_change", channel=0, control=10, value=64)
    result = mc.process_midi_message(msg)
    assert isinstance(result, dict)
    assert result["type"] == "CC(7bit)"
    assert result["CC number"] == 10
    assert 0 <= result["value"] <= 127

    # also touch a static helper to ensure module import is healthy
    scaled = MidiController.scale_value(8192, 0.0, 127.0)
    assert 0.0 <= scaled <= 127.0
