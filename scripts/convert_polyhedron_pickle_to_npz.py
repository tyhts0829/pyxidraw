#!/usr/bin/env python3
"""
Polyhedron pickle -> NPZ converter

Converts data/regular_polyhedron/*_vertices_list.pkl into
*_vertices_list.npz with arrays saved as arr_0, arr_1, ... (float32).

Usage:
  python scripts/convert_polyhedron_pickle_to_npz.py [--delete-original] [--dry-run] [--verbose]

Notes:
  - Safe to re-run. Existing .npz files are skipped unless --force is set.
  - The script only touches files under data/regular_polyhedron.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def find_pickle_files(base: Path) -> list[Path]:
    return sorted(base.glob("*_vertices_list.pkl"))


def load_pickle(path: Path) -> list[np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    # Accept list/tuple of arrays-like
    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"{path.name}: expected list/tuple, got {type(obj).__name__}")
    arrays: list[np.ndarray] = []
    for i, a in enumerate(obj):
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim == 1:
            # Allow flat coordinates; try to reshape to (-1, 3)
            if arr.size % 3 != 0:
                raise ValueError(f"{path.name}: item {i} cannot reshape to (-1, 3)")
            arr = arr.reshape(-1, 3)
        elif arr.ndim != 2:
            raise ValueError(f"{path.name}: item {i} has invalid ndim={arr.ndim}")
        arrays.append(arr)
    return arrays


def save_npz(arrays: Iterable[np.ndarray], out_path: Path, *, force: bool = False) -> None:
    if out_path.exists() and not force:
        return
    payload = {f"arr_{i}": np.asarray(a, dtype=np.float32) for i, a in enumerate(arrays)}
    np.savez(out_path, **payload)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "regular_polyhedron"

    parser = argparse.ArgumentParser(description="Convert polyhedron pickle data to NPZ format.")
    parser.add_argument("--delete-original", action="store_true", help="Delete .pkl after successful conversion")
    parser.add_argument("--dry-run", action="store_true", help="Only report actions; do not write files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npz files")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if not data_dir.exists():
        print(f"No directory: {data_dir}")
        return 1

    pkl_files = find_pickle_files(data_dir)
    if not pkl_files:
        print("No pickle files to convert.")
        return 0

    converted = 0
    for pkl in pkl_files:
        out = pkl.with_suffix(".npz")
        if args.verbose:
            print(f"Converting {pkl.name} -> {out.name}")

        try:
            arrays = load_pickle(pkl)
        except Exception as e:
            print(f"[SKIP] {pkl.name}: {e}")
            continue

        if args.dry_run:
            converted += 1
            continue

        try:
            save_npz(arrays, out, force=args.force)
            converted += 1
            if args.delete_original and out.exists():
                pkl.unlink(missing_ok=True)
                if args.verbose:
                    print(f"Deleted {pkl.name}")
        except Exception as e:
            print(f"[ERROR] {pkl.name}: {e}")

    print(f"Converted {converted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

