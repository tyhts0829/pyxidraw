"""
3Dプリンタをペンプロッターとして使用するためのGcodeを生成するクラス

G00: 位置決め: 指定した位置へ早送りで移動
G01: 直線補間: 指定した位置へ直線補間で移動
G28: オートホーム: ホーム位置へ移動
G90: 絶対座標: 絶対座標モードを指定
G92: 位置リセット: 現在の位置を原点とする
M107: ファンオフ: ファンをオフにする
M117: メッセージ表示: LCDにメッセージを表示する
M300: ビープ音: ビープ音を鳴らす

"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TODO y座標の逆転 y_machine = canvas_size.y - y_virtual


@dataclass
class GcodeConfig:
    """Gcode生成設定"""

    save_dir: Path = Path("output")
    offsets: Tuple[float, float] = (91, -0.75)
    z_up: float = 3.0
    z_down: float = -2.0
    draw_speed: int = 1000
    travel_speed: int = 1500
    connect_distance: float = 0.5
    printer_range: Tuple[float, float] = (0.0, 300.0)
    a4_width_mm: float = 210.0


class GcodeGenerator:
    def __init__(self, config: Optional[GcodeConfig] = None) -> None:
        self.config = config or GcodeConfig()
        self.config.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_name: Optional[str] = None
        self.canvas_size: Optional[Tuple[float, float]] = None
        self.vertices_list: Optional[List[np.ndarray]] = None

    def _load_pkl(self, file_path: Optional[Path] = None) -> None:
        """pklファイルを読み込む

        Args:
            file_path: 読み込むファイルのパス。Noneの場合はGUIダイアログを表示
        """
        if file_path is None:
            # GUI モード
            root = Tk()
            root.wm_attributes("-topmost", 1)
            root.lift()
            root.focus_force()
            root.withdraw()
            file_path_str = filedialog.askopenfilename(
                parent=root, title="ファイルを選択", filetypes=[("vertices_list pkl file", "*.pkl")]
            )
            if not file_path_str:
                raise ValueError("No file selected")
            file_path = Path(file_path_str)

        self.save_name = file_path.stem + ".gcode"
        # pklを開く
        with open(file_path, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)
        self.canvas_size = data["canvas_size"]
        self.vertices_list = data["vertices_list"]

    def adj_x(self, vertices_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        A4紙とは別の紙を使う場合のためにx座標を調整する
        """
        x_paper: float = self.canvas_size[0]  # type: ignore
        adj: float = self.config.a4_width_mm - x_paper  # /2 しないとばっちり中心が揃った
        new_vertices_list: List[np.ndarray] = []
        for vertices in vertices_list:
            vertices_copy = vertices.astype(float)
            vertices_copy[:, 0] += adj
            new_vertices_list.append(vertices_copy)
        return new_vertices_list

    def invert_y(self, vertices_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        画面の座標は左上が0, 0だが、3Dプリンタの座標は左下が0, 0なのでy座標を逆転する
        """
        new_vertices_list: List[np.ndarray] = []
        for vertices in vertices_list:
            vertices_copy = vertices.astype(float)
            vertices_copy[:, 1] = self.canvas_size[1] - vertices_copy[:, 1]  # type: ignore
            new_vertices_list.append(vertices_copy)
        return new_vertices_list

    def generate_gcode(self, file_path: Optional[Path] = None) -> None:
        """Gcodeを生成する

        Args:
            file_path: 読み込むpklファイルのパス。Noneの場合はGUIダイアログを表示
        """
        self._load_pkl(file_path)
        if self.vertices_list is None:
            raise ValueError("vertices_list is not loaded. _load_pkl() failed.")
        vertices_list = self._drop_z(self.vertices_list)
        vertices_list = self.optimize_vertices_list(vertices_list)
        vertices_list = self.adj_x(vertices_list)
        vertices_list = self.invert_y(vertices_list)
        self.assert_range(vertices_list)

        gcode: List[str] = []
        gcode.extend(self._generate_header())
        gcode.extend(self._generate_body(vertices_list))
        gcode.extend(self._generate_footer())
        self._save_gcode(gcode)

    def assert_range(self, vertices_list: List[np.ndarray]) -> None:
        """
        座標が3Dプリンタの範囲内に収まっているか確認
        """
        range_min, range_max = self.config.printer_range
        for vertices in vertices_list:
            for vertex in vertices:
                if not (
                    range_min <= vertex[0] <= range_max and range_min <= vertex[1] <= range_max
                ):
                    raise ValueError(f"vertex: {vertex} is out of range [{range_min}, {range_max}]")

    def _drop_z(self, vertices_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        verticesのz座標を削除
        """
        return [vertices[:, :2] for vertices in vertices_list]

    def optimize_vertices_list(self, vertices2d_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        描画の終点と次の始点の距離がCONNECT_DISTANCE以下ならつなげて描画する
        """
        new_vertices2d_list: List[np.ndarray] = []
        end_point: Optional[np.ndarray] = None
        next_start_point: Optional[np.ndarray] = None
        for i, vertices2d in enumerate(vertices2d_list):
            end_point = vertices2d[-1]
            if i + 1 < len(vertices2d_list):
                next_start_point = vertices2d_list[i + 1][0]
            else:
                next_start_point = None
            if (
                next_start_point is not None
                and np.linalg.norm(end_point - next_start_point) < self.config.connect_distance
            ):
                new_vertices2d_list.append(np.vstack((vertices2d, vertices2d_list[i + 1])))
            else:
                new_vertices2d_list.append(vertices2d)
        return new_vertices2d_list

    def _generate_header(self) -> List[str]:  # TODO 現行の設定を確認してみる
        header: List[str] = []
        header.append("; ====== Header ======")
        header.append("G21 ; Set units to millimeters")
        header.append("G90 ; Absolute positioning")
        header.append("G28 ; Home all axes")
        header.append("M107 ; Turn off fan")
        header.append("M420 S1 Z10; Enable bed leveling matrix")
        return header

    def _generate_body(self, vertices_list: List[np.ndarray]) -> List[str]:
        body: List[str] = []
        body.append("; ====== Body ======")
        for n_lines, vertices in enumerate(vertices_list):
            body.append(f"; line {n_lines} start")
            for n_vertex, vertex in enumerate(vertices):
                if n_vertex == 0:  # 最初の点は早送りで移動
                    body.extend(self._prepare_travel())
                elif n_vertex == 1:  # 2点目以降は直線補間で移動
                    body.extend(self._prepare_draw())
                x: float
                y: float
                x, y = vertex
                x += self.config.offsets[0]
                y += self.config.offsets[1]
                body.append(f"G1 X{x} Y{y}")
            body.append(f"; line {n_lines} end")
        return body

    def _generate_footer(self) -> List[str]:  # TODO 現行の設定を確認してみる
        footer: List[str] = []
        footer.append("; ====== Footer ======")
        footer.append("G01 Z30 ; Move the pen up")
        return footer

    def _prepare_draw(self) -> List[str]:
        gcode: List[str] = []
        gcode.append(f"G1 Z{self.config.z_down}")
        gcode.append(f"G1 F{self.config.draw_speed}")
        return gcode

    def _prepare_travel(self) -> List[str]:
        gcode: List[str] = []
        gcode.append(f"G1 Z{self.config.z_up}")
        gcode.append(f"G1 F{self.config.travel_speed}")
        return gcode

    def _save_gcode(self, gcode: List[str]) -> None:
        if self.save_name is None:
            raise ValueError("save_name is not set. Call _load_pkl() first.")
        with open(self.config.save_dir / self.save_name, "w") as f:
            f.write("\n".join(gcode))


if __name__ == "__main__":
    gg = GcodeGenerator()
    gg.generate_gcode()
