"""
Convert legacy pickle assets to NPZ (Proposal 6).

Usage:
    python scripts/convert_pickle_assets.py

It scans data/regular_polyhedron and data/sphere for *_vertices_list.pkl
and writes corresponding *_vertices_list.npz files.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np


def convert_folder(folder: Path) -> List[Path]:
    written: List[Path] = []
    for pkl in folder.glob("*_vertices_list.pkl"):
        npz = pkl.with_suffix(".npz")
        if npz.exists():
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        # Data can be a list of arrays or list-like
        arrays = [np.asarray(a, dtype=np.float32) for a in data]
        # Save as numbered arrays for simplicity
        np.savez(npz, **{f"arr_{i}": a for i, a in enumerate(arrays)})
        written.append(npz)
    return written


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [root / "data" / "regular_polyhedron", root / "data" / "sphere"]
    for t in targets:
        if not t.exists():
            continue
        written = convert_folder(t)
        if written:
            logging.getLogger(__name__).info("Converted %d files in %s", len(written), t)


if __name__ == "__main__":
    main()
