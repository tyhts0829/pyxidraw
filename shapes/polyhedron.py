from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseShape
from engine.core.geometry import Geometry
from .registry import shape


@shape
class Polyhedron(BaseShape):
    """Regular polyhedron shape generator using pre-computed vertex data."""
    
    _vertices_cache = None
    
    # Polyhedron type mapping
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
        """Load pre-computed polyhedron vertex data."""
        if cls._vertices_cache is None:
            cls._vertices_cache = {}
            data_dir = Path(__file__).parents[1] / "data" / "regular_polyhedron"

            # Check if data directory exists
            if not data_dir.exists():
                cls._vertices_cache = None
                return

            polyhedrons = ["tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron"]
            for polyhedron in polyhedrons:
                # Prefer new npz format
                npz_file = data_dir / f"{polyhedron}_vertices_list.npz"
                if npz_file.exists():
                    with np.load(npz_file) as data:
                        # Accept either 'arr_i' keys or 'arrays' list
                        if 'arrays' in data.files:
                            arrays = data['arrays']
                            cls._vertices_cache[polyhedron] = [np.array(a, dtype=np.float32) for a in arrays]
                        else:
                            # Gather arr_0, arr_1... in order
                            keys = sorted([k for k in data.files if k.startswith('arr_')], key=lambda k: int(k.split('_')[1]))
                            cls._vertices_cache[polyhedron] = [data[k].astype(np.float32) for k in keys]
                    continue

                # Legacy pickle (deprecated). Kept for transitional use.
                pkl_file = data_dir / f"{polyhedron}_vertices_list.pkl"
                if pkl_file.exists():
                    with open(pkl_file, "rb") as f:
                        cls._vertices_cache[polyhedron] = pickle.load(f)
    
    def generate(self, polygon_type: str | int = "tetrahedron", **params: Any) -> Geometry:
        """Generate a regular polyhedron.
        
        Args:
            polygon_type: Type of polyhedron (name or number of faces)
            **params: Additional parameters (ignored)
            
        Returns:
            Geometry object containing polyhedron edges
        """
        if polygon_type not in self._TYPE_MAP:
            raise ValueError(f"Invalid polygon_type: {polygon_type}")
        
        shape_name = self._TYPE_MAP[polygon_type]
        
        # Try to load pre-computed data
        self._load_vertices_data()
        
        if self._vertices_cache and shape_name in self._vertices_cache:
            vertices_list = self._vertices_cache[shape_name]
            # Convert to list of numpy arrays if needed
            if isinstance(vertices_list, list):
                converted_list = [np.array(v, dtype=np.float32) for v in vertices_list]
                return Geometry.from_lines(converted_list)
            return Geometry.from_lines(vertices_list)
        
        # Fallback: generate simple polyhedron
        return Geometry.from_lines(self._generate_simple_polyhedron(shape_name))
    
    def _generate_simple_polyhedron(self, shape_name: str) -> list[np.ndarray]:
        """Generate simple polyhedron vertices."""
        if shape_name == "tetrahedron":
            # Simple tetrahedron
            vertices = np.array([
                [0, 0, 0.5],
                [0.433, 0, -0.25],
                [-0.216, 0.375, -0.25],
                [-0.216, -0.375, -0.25]
            ], dtype=np.float32)
            
            # Edge connections
            edges = [
                [vertices[0], vertices[1]],
                [vertices[0], vertices[2]],
                [vertices[0], vertices[3]],
                [vertices[1], vertices[2]],
                [vertices[2], vertices[3]],
                [vertices[3], vertices[1]]
            ]
            return [np.array(edge, dtype=np.float32) for edge in edges]
        
        elif shape_name == "hexahedron" or shape_name == "cube":
            # Simple cube
            d = 0.5
            vertices = np.array([
                [-d, -d, -d], [d, -d, -d], [d, d, -d], [-d, d, -d],
                [-d, -d, d], [d, -d, d], [d, d, d], [-d, d, d]
            ], dtype=np.float32)
            
            # Edge connections
            edges = []
            # Bottom face
            for i in range(4):
                edges.append([vertices[i], vertices[(i + 1) % 4]])
            # Top face  
            for i in range(4):
                edges.append([vertices[i + 4], vertices[((i + 1) % 4) + 4]])
            # Vertical edges
            for i in range(4):
                edges.append([vertices[i], vertices[i + 4]])
                
            return [np.array(edge, dtype=np.float32) for edge in edges]
        
        elif shape_name == "octahedron":
            # Simple octahedron
            vertices = np.array([
                [0.5, 0, 0], [-0.5, 0, 0],
                [0, 0.5, 0], [0, -0.5, 0],
                [0, 0, 0.5], [0, 0, -0.5]
            ], dtype=np.float32)
            
            # Edge connections
            edges = []
            # Connect top vertex to middle square
            for i in range(4):
                edges.append([vertices[4], vertices[i]])
            # Connect bottom vertex to middle square
            for i in range(4):
                edges.append([vertices[5], vertices[i]])
            # Middle square
            edges.append([vertices[0], vertices[2]])
            edges.append([vertices[2], vertices[1]])
            edges.append([vertices[1], vertices[3]])
            edges.append([vertices[3], vertices[0]])
            
            return [np.array(edge, dtype=np.float32) for edge in edges]
        
        else:
            # Default to tetrahedron for unsupported types
            return self._generate_simple_polyhedron("tetrahedron")
