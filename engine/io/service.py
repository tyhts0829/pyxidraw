from __future__ import annotations

from typing import Dict, Mapping

from ..core.tickable import Tickable
from .manager import MidiControllerManager


class MidiService(Tickable):
    """MIDI デバイスのポーリングと CC スナップショット取得を司るサブシステム。

    Canvas や Worker など他サブシステムからは “現在の CC 値を辞書で渡す供給源”
    として扱われる。`tick()` は毎フレーム呼び出され、内部で
    `MidiControllerManager.update_midi_controllers()` を実行するだけなので軽い。

    Examples
    --------
    ```python
    midi_manager = connect_midi_controllers()
    midi_subsys  = MidiSubsystem(midi_manager)
    worker       = WorkerSubsystem(
        fps=60,
        draw_callback=draw_callback,
        cc_supplier=midi_subsys.snapshot,   # ← ここに渡す
    )
    ```
    """

    # ------------------------------------------------------------------ #
    # コンストラクタ                                                     #
    # ------------------------------------------------------------------ #
    def __init__(self, controller_manager: MidiControllerManager) -> None:
        """
        Args
        ----
        controller_manager:
            既に接続済みの ``MidiControllerManager`` インスタンス。
        """
        self._manager = controller_manager
        self._latest_cc_flat_cache: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Tickable interface                                                 #
    # ------------------------------------------------------------------ #
    def tick(self, dt: float) -> None:  # noqa: D401
        """1 フレーム分のポーリングを行い CC 値を更新する。"""
        self._manager.update_midi_controllers()
        # 必要に応じてキャッシュを更新（lazy でも OK）
        self._latest_cc_flat_cache = self._flatten_cc_values()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Mapping[int, int]:
        """“最新 CC 値” のイミュータブルスナップショットを返す。

        WorkerSubsystem の ``cc_supplier`` にそのまま渡せる形
        ``{cc_number: value}`` を返す。辞書は *浅いコピー* なので
        呼び出し側で自由に破壊的変更しても安全。
        """
        # copy() でイミュータブル保証
        return dict(self._latest_cc_flat_cache)

    def save_all_cc(self) -> None:
        """各コントローラの CC をファイルに永続化する。"""
        self._manager.save_cc()

    # ------------------------------------------------------------------ #
    # 内部ユーティリティ                                                 #
    # ------------------------------------------------------------------ #
    def _flatten_cc_values(self) -> Dict[int, int]:
        """複数コントローラの CC 値を “CC 番号ベース” に 1 辞書へ統合。

        - **同じ CC 番号が複数デバイスで競合する場合**
          先に検出したデバイスを優先する（後勝ちにしたい場合は `reversed()`）。
        - デバイスごとに完全に独立させたい場合は、このメソッドを
          オーバーライドした派生クラスを作ると良い。
        self._manager.controllersはマルチスレッド環境での共有リソースではないので、スレッドセーフ対策は不要
        """
        flat: Dict[int, int] = {}
        for controller in self._manager.controllers.values():
            for cc_num, value in controller.cc.items():
                if cc_num not in flat:
                    flat[cc_num] = value
        return flat
