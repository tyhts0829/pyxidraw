from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .base import BaseShape
from .registry import shape


@shape
class Polyhedron(BaseShape):
    """事前計算済みの頂点データを用いて正多面体の線分群を生成するシェイプ。"""

    _vertices_cache = None

    # 多面体タイプのマッピング
    _TYPE_MAP = {
        "tetrahedron": "tetrahedron",
        4: "tetrahedron",
        "tetra": "tetrahedron",
        "hexahedron": "hexahedron",
        6: "hexahedron",
        "hexa": "hexahedron",
        "cube": "hexahedron",
        "box": "hexahedron",
        "octahedron": "octahedron",
        8: "octahedron",
        "octa": "octahedron",
        "dodecahedron": "dodecahedron",
        12: "dodecahedron",
        "dodeca": "dodecahedron",
        "icosahedron": "icosahedron",
        20: "icosahedron",
        "icosa": "icosahedron",
    }

    @classmethod
    def _load_vertices_data(cls):
        """事前計算済みの多面体頂点データ（.npz のみ）を読み込みます。"""
        if cls._vertices_cache is None:
            cls._vertices_cache = {}
            data_dir = Path(__file__).parents[1] / "data" / "regular_polyhedron"

            # データディレクトリの存在を確認
            if not data_dir.exists():
                cls._vertices_cache = None
                return

            polyhedrons = ["tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron"]
            for polyhedron in polyhedrons:
                # 正式な npz 形式
                npz_file = data_dir / f"{polyhedron}_vertices_list.npz"
                if npz_file.exists():
                    with np.load(npz_file) as data:
                        # 'arr_i' 形式の複数配列キーまたは 'arrays' リストのどちらにも対応
                        if "arrays" in data.files:
                            arrays = data["arrays"]
                            cls._vertices_cache[polyhedron] = [
                                np.array(a, dtype=np.float32) for a in arrays
                            ]
                        else:
                            # arr_0, arr_1... を順序通りに収集
                            keys = sorted(
                                [k for k in data.files if k.startswith("arr_")],
                                key=lambda k: int(k.split("_")[1]),
                            )
                            cls._vertices_cache[polyhedron] = [
                                data[k].astype(np.float32) for k in keys
                            ]
                    continue

    def generate(self, polygon_type: str | int = "tetrahedron", **params: Any) -> Geometry:
        """正多面体を生成します。

        引数:
            polygon_type: 多面体の種類（名前または面数）
            **params: 追加パラメータ（未使用）

        返り値:
            多面体のエッジ群を含む Geometry
        """
        if polygon_type not in self._TYPE_MAP:
            raise ValueError(f"polygon_type が不正です: {polygon_type}")

        shape_name = self._TYPE_MAP[polygon_type]

        # 事前計算データの読み込みを試行
        self._load_vertices_data()

        if self._vertices_cache and shape_name in self._vertices_cache:
            vertices_list = self._vertices_cache[shape_name]
            # 必要に応じて numpy 配列のリストへ変換
            if isinstance(vertices_list, list):
                converted_list = [np.array(v, dtype=np.float32) for v in vertices_list]
                return Geometry.from_lines(converted_list)
            return Geometry.from_lines(vertices_list)

        # フォールバック: 簡易な多面体を生成
        return Geometry.from_lines(self._generate_simple_polyhedron(shape_name))

    def _generate_simple_polyhedron(self, shape_name: str) -> list[np.ndarray]:
        """簡易な多面体の頂点・辺リストを生成します。"""
        if shape_name == "tetrahedron":
            # 単純な四面体
            vertices = np.array(
                [[0, 0, 0.5], [0.433, 0, -0.25], [-0.216, 0.375, -0.25], [-0.216, -0.375, -0.25]],
                dtype=np.float32,
            )

            # 辺の接続定義
            edges = [
                [vertices[0], vertices[1]],
                [vertices[0], vertices[2]],
                [vertices[0], vertices[3]],
                [vertices[1], vertices[2]],
                [vertices[2], vertices[3]],
                [vertices[3], vertices[1]],
            ]
            return [np.array(edge, dtype=np.float32) for edge in edges]

        elif shape_name == "hexahedron" or shape_name == "cube":
            # 単純な立方体
            d = 0.5
            vertices = np.array(
                [
                    [-d, -d, -d],
                    [d, -d, -d],
                    [d, d, -d],
                    [-d, d, -d],
                    [-d, -d, d],
                    [d, -d, d],
                    [d, d, d],
                    [-d, d, d],
                ],
                dtype=np.float32,
            )

            # 辺の接続定義
            edges = []
            # 底面
            for i in range(4):
                edges.append([vertices[i], vertices[(i + 1) % 4]])
            # 上面
            for i in range(4):
                edges.append([vertices[i + 4], vertices[((i + 1) % 4) + 4]])
            # 垂直の辺
            for i in range(4):
                edges.append([vertices[i], vertices[i + 4]])

            return [np.array(edge, dtype=np.float32) for edge in edges]

        elif shape_name == "octahedron":
            # 単純な八面体
            vertices = np.array(
                [[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]],
                dtype=np.float32,
            )

            # 辺の接続定義
            edges = []
            # 上側頂点と中間の四角形を接続
            for i in range(4):
                edges.append([vertices[4], vertices[i]])
            # 下側頂点と中間の四角形を接続
            for i in range(4):
                edges.append([vertices[5], vertices[i]])
            # 中央の四角形
            edges.append([vertices[0], vertices[2]])
            edges.append([vertices[2], vertices[1]])
            edges.append([vertices[1], vertices[3]])
            edges.append([vertices[3], vertices[0]])

            return [np.array(edge, dtype=np.float32) for edge in edges]

        else:
            # 未対応タイプは四面体にフォールバック
            return self._generate_simple_polyhedron("tetrahedron")
