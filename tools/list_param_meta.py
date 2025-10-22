"""全 shape/effect の `__param_meta__` を一覧する開発用スクリプト。

前提: 事前に `pip install -e .[dev]` を実行し、`api`/`effects` などを import 可能にしておく。
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping


def _normalize_meta(meta: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(meta, Mapping):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for key, value in meta.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        result[key] = value
    return result


def _collect_shapes() -> dict[str, dict[str, Mapping[str, Any]]]:
    import shapes  # noqa: F401
    from shapes.registry import get_shape, list_shapes

    entries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name in list_shapes():
        fn = get_shape(name)
        entries[name] = _normalize_meta(getattr(fn, "__param_meta__", None))
    return entries


def _collect_effects() -> dict[str, dict[str, Mapping[str, Any]]]:
    import effects  # noqa: F401
    from effects.registry import get_effect, list_effects

    entries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name in list_effects():
        fn = get_effect(name)
        entries[name] = _normalize_meta(getattr(fn, "__param_meta__", None))
    return entries


def _format_mapping(data: Mapping[str, Mapping[str, Any]]) -> str:
    if not data:
        return "<no meta>"
    parts: list[str] = []
    for key in sorted(data):
        value = data[key]
        summary = ", ".join(f"{k}={v!r}" for k, v in sorted(value.items()))
        parts.append(f"{key}: {summary}")
    return "; ".join(parts)


def _print_human_readable(sections: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    for section_name, entries in sections.items():
        print(section_name)
        for name in sorted(entries):
            formatted = _format_mapping(entries[name])
            print(f"  - {name}: {formatted}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="List parameter metadata for shapes and effects.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    sections = {
        "Shapes": _collect_shapes(),
        "Effects": _collect_effects(),
    }

    if args.json:
        print(json.dumps(sections, indent=2, ensure_ascii=False))
        return

    _print_human_readable(sections)


if __name__ == "__main__":
    main()
