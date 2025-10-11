"""
どこで: `engine.io` の管理層。
何を: 複数 `MidiController` の登録/更新/保存と、デバイス検出 `connect_midi_controllers()`。
なぜ: 設定に基づくコントローラ管理を一元化し、ポーリング更新の窓口を提供するため。
"""

from util.utils import load_config

from .controller import MidiController


class MidiControllerManager:
    """
    MidiControllerに名前をつけて管理するクラス。
    connect_midi_controllers()でインスタンス化する。
    """

    def __init__(self):
        self.controllers = {}

    def append_controller(self, controller_name, controller):
        self.controllers[controller_name] = controller

    def update_midi_controllers(self):
        for controller in self.controllers.values():
            for msg in controller.iter_pending():
                controller.update_cc(msg)

    def save_cc(self):
        for controller in self.controllers.values():
            controller.save_cc()

    def __getitem__(self, controller_name):
        return self.controllers[controller_name]

    @property
    def controller_names(self):
        return list(self.controllers.keys())

    def __repr__(self):
        result = "MidiControllerManager:\n"
        for controller_name, controller in self.controllers.items():
            result += f"  {controller}\n"
        return result


def connect_midi_controllers() -> MidiControllerManager:
    """
    PCに接続されているMIDIデバイスを検出し、MidiControllerオブジェクトにしてMidiControllerManagerに格納して返す。
    """
    # ローカル import（未導入時は ImportError がそのまま伝搬）
    import mido  # type: ignore

    cfg = load_config() or {}
    devices = cfg.get("midi_devices", []) if isinstance(cfg, dict) else []

    def find_connected_port_name(port_name):
        """
        指定されたデバイス名を持つポートを検索し、見つかったらそのポート名を返す。
        """
        try:
            return [port for port in mido.get_input_names() if port_name in port][0]  # type: ignore
        except IndexError:
            return None

    midi_controller_manager = MidiControllerManager()
    for device in devices:
        port_name = find_connected_port_name(device["port_name"])
        if port_name:
            midi_controller = MidiController(
                port_name, cc_map=device["cc_map"], mode=device["mode"]
            )
            midi_controller_manager.append_controller(device["controller_name"], midi_controller)
    return midi_controller_manager


if __name__ == "__main__":
    connect_midi_controllers()
