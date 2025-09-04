import pickle
from pathlib import Path

import numpy as np
import trimesh
import logging

from effects.effector import apply_effectors

####
# 直径1の球のデータ生成
####


def generate_vertices_list(vertices, edges):
    vertices_list = []
    for edge in edges:
        vertices_list.append(np.array([vertices[edge[0]], vertices[edge[1]]]))
    return vertices_list


def generate_shpere_mesh_tri(subdivisions):
    mesh = trimesh.primitives.Sphere(radius=0.5, subdivisions=subdivisions)
    edges = mesh.edges_unique  # エッジは頂点インデックスのペアで表現
    vertices = mesh.vertices  # 頂点座標を取得
    return generate_vertices_list(vertices, edges)


def generate_shpere_mesh_circle(subdivisions):
    # def generate_circle_mesh
    pass


def generate_data():
    subdivisions = list(range(8))
    data = {}
    for subdivision in subdivisions:
        data[subdivision] = generate_shpere_mesh_tri(subdivision)
    return data


if __name__ == "__main__":
    SAVE_DIR = Path(r"data/sphere")
    data = generate_data()
    for subdivision, vertices_list in data.items():
        save_name = f"sphere_tri_{subdivision}_vertices_list.pkl"
        with open(SAVE_DIR / save_name, "wb") as f:
            pickle.dump(vertices_list, f)
        logging.getLogger(__name__).info("saved: %s", SAVE_DIR / save_name)
    logging.getLogger(__name__).info("finish")
