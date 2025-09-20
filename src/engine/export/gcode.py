"""
どこで: `engine.export.gcode`。
何を: G-code 出力のパラメータ定義と書き出しクラスのスケルトンを提供する。
なぜ: ランタイムから非ブロッキングで G-code を保存するための土台を分離するため。

本段階（Stage 0/3）では変換ロジックは未実装で、空の本体を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Tuple

import numpy as np


@dataclass(frozen=True)
class GCodeParams:
    """G-code 生成パラメータ。

    属性:
        travel_feed: ペンアップ移動のフィードレート [mm/min]。
        draw_feed: ペンダウン描画のフィードレート [mm/min]。
        z_up: ペンアップ時の Z 高さ [mm]。
        z_down: ペンダウン時の Z 高さ [mm]。
        y_down: True で Y 軸下向き座標系に変換する（既定は False）。
        origin: 出力座標の原点 [mm]（X, Y）。
        decimals: 小数点以下の桁数（出力の丸め）。
    """

    travel_feed: float
    draw_feed: float
    z_up: float
    z_down: float
    y_down: bool
    origin: Tuple[float, float]
    decimals: int = 3


class GCodeWriter:
    """G-code 書き出しクラス（スケルトン）。

    `coords` と `offsets` から 2D パスを走査し、`fp` へ G-code をテキスト出力する。
    本段階では未実装。将来、直線補間のみを前提として実装する。
    """

    def write(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        params: GCodeParams,
        fp: IO[str],
    ) -> None:
        """与えられたジオメトリを G-code として `fp` に書き出す。

        引数:
            coords: 形状座標の連結配列。形状は "offsets" で区切られる想定。
            offsets: 各ラインの開始インデックス群（終端を含まない累積オフセット）。
            params: G-code 出力パラメータ。
            fp: テキスト書き出し先（開かれたファイルオブジェクト）。
        """
        # Stage 0/3: 未実装のため将来実装で置換する
        raise NotImplementedError("GCodeWriter.write is not implemented yet")
