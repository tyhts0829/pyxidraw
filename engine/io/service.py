from __future__ import annotations

import logging
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
        self._latest_cc_flat_cache: Dict[int, float] = {}
        self._logger = logging.getLogger(__name__)

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
    def snapshot(self) -> Mapping[int, float]:
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
    def _flatten_cc_values(self) -> Dict[int, float]:
        """複数コントローラの CC 値を “CC 番号ベース” に 1 辞書へ統合。

        - **同じ CC 番号が複数デバイスで競合する場合**
          先に検出したデバイスを優先する（後勝ちにしたい場合は `reversed()`）。
        - デバイスごとに完全に独立させたい場合は、このメソッドを
          オーバーライドした派生クラスを作ると良い。
        self._manager.controllersはマルチスレッド環境での共有リソースではないので、スレッドセーフ対策は不要
        """
        flat: Dict[int, float] = {}
        for controller in self._manager.controllers.values():
            # DualKeyDictの場合は整数キーのみアクセス可能
            # 文字列キーから対応する整数キーを取得して値を設定
            if hasattr(controller.cc, '_str_to_value') and hasattr(controller.cc, '_reverse_cc_map'):
                # 文字列キーから整数キーと値を取得
                self._logger.debug("_str_to_value keys: %s", list(controller.cc._str_to_value.keys()))
                self._logger.debug("_reverse_cc_map: %s", controller.cc._reverse_cc_map)
                for str_key, value in controller.cc._str_to_value.items():
                    int_key = controller.cc._reverse_cc_map.get(str_key)
                    self._logger.debug("Looking up '%s' -> %s", str_key, int_key)
                    if int_key is not None:
                        flat[int_key] = value
                        self._logger.debug("Flattened CC#%s (from %s): %s", int_key, str_key, value)
                    else:
                        self._logger.debug("No mapping found for '%s' (value: %s)", str_key, value)
            else:
                # 通常の辞書の場合
                for cc_num, value in controller.cc.items():
                    flat[cc_num] = value
                    self._logger.debug("Flattened CC#%s: %s", cc_num, value)
        return flat
