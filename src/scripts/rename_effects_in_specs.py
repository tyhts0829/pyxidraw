#!/usr/bin/env python3
"""
Spec renamer: migrate effect names and selected params from old → new naming.

Usage:
  - Read spec JSON from stdin and write migrated JSON to stdout:
      cat spec.json | python scripts/rename_effects_in_specs.py > out.json
  - Or pass file paths:
      python scripts/rename_effects_in_specs.py in.json out.json

This tool is idempotent and only rewrites known names/params.
"""
import json
import sys

NAME_MAP = {
    "translation": "translate",
    "rotation": "rotate",
    "scaling": "scale",
    "transform": "affine",
    "noise": "displace",
    "filling": "fill",
    "array": "repeat",
    "buffer": "offset",
    "subdivision": "subdivide",
    "trimming": "trim",
    "dashify": "dash",
    "wave": "ripple",
    "webify": "weave",
}


def migrate_params(name: str, params: dict) -> dict:
    p = dict(params)
    if name == "translate":
        if all(k in p for k in ("offset_x", "offset_y", "offset_z")) and "delta" not in p:
            p["delta"] = (p.pop("offset_x"), p.pop("offset_y"), p.pop("offset_z"))
    elif name == "rotate":
        if "rotate" in p and all(isinstance(x, (int, float)) for x in p["rotate"]):
            p["angles_rad"] = [float(v) * 2 * 3.141592653589793 for v in p.pop("rotate")]
    elif name == "fill":
        if "pattern" in p and "mode" not in p:
            p["mode"] = p.pop("pattern")
        if "angle" in p and "angle_rad" not in p:
            p["angle_rad"] = p.pop("angle")
    elif name == "repeat":
        if "n_duplicates" in p and "count" not in p:
            try:
                count = int(round(float(p["n_duplicates"]) * 10))
                p["count"] = count
            except Exception:
                pass
    elif name == "displace":
        if "intensity" in p and "amplitude_mm" not in p:
            p["amplitude_mm"] = p["intensity"]
        if "frequency" in p and "spatial_freq" not in p:
            p["spatial_freq"] = p["frequency"]
        if "time" in p and "t_sec" not in p:
            p["t_sec"] = p["time"]
    return p


def migrate_spec(spec):
    out = []
    for step in spec:
        name = str(step.get("name"))
        params = dict(step.get("params", {}))
        new_name = NAME_MAP.get(name, name)
        out.append(
            {
                "name": new_name,
                "params": migrate_params(new_name, params),
            }
        )
    return out


def main():
    if len(sys.argv) == 3:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            spec = json.load(f)
        out = migrate_spec(spec)
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return 0
    data = sys.stdin.read()
    spec = json.loads(data)
    out = migrate_spec(spec)
    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
