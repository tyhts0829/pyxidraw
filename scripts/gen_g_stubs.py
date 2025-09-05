"""
Generate `api/__init__.pyi` to expose shape names on `G` for IDE autocompletion.

Usage:
    python -m scripts.gen_g_stubs

Notes:
    - Reads registered shapes from `api.shape_registry.list_registered_shapes()`.
    - Writes a Protocol `_GShapes` with attributes `<shape>: Callable[..., Geometry]`.
    - Re-exports public API (`G`, `E`, `Geometry`, etc.) to keep type checkers happy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any, get_type_hints, get_origin, get_args
import inspect
import re
import sys
import types


def _is_valid_identifier(name: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None


def _format_type(tp: Any) -> str:
    try:
        s = str(tp)
    except Exception:
        return "Any"
    # Normalize common typing representations
    s = s.replace("typing.", "")
    # Builtin classes like <class 'int'> -> int
    s = re.sub(r"^<class '([a-zA-Z_][a-zA-Z0-9_\.]*)'>$", r"\1", s)
    return s


def _extract_param_docs(gen_obj: Any) -> tuple[str | None, dict[str, str]]:
    """Extract a short summary and per-parameter docs from a `generate` docstring.

    Returns (summary, {param: short_desc}). Very defensive and lenient:
    - Supports both English "Args:" and Japanese "引数:" headers.
    - Considers continuation lines as part of the last seen parameter.
    - Shortens each description to a single concise line (no trailing punctuation).
    """
    doc = inspect.getdoc(gen_obj) or ""
    if not doc:
        return None, {}

    lines = doc.splitlines()
    # Summary: first non-empty line that is not a section header
    summary: str | None = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith(("args:", "returns:", "raises:", "引数:", "返り値:", "例:")):
            break
        summary = s
        break

    # Extract Args block
    in_args = False
    last_param: str | None = None
    param_docs: dict[str, str] = {}

    def _shorten(text: str, limit: int = 120) -> str:
        t = " ".join(text.split())  # collapse whitespace
        # Prefer to keep periods to avoid breaking decimals like 0.5
        # For Japanese full stop, trim at the first occurrence to keep it concise.
        for delim in ["。", "．"]:
            if delim in t:
                t = t.split(delim)[0]
                break
        if len(t) > limit:
            t = t[: limit - 1] + "…"
        return t

    for ln in lines:
        s = ln.rstrip()
        s_stripped = s.strip()
        lower = s_stripped.lower()
        if lower.startswith("args:") or s_stripped.startswith("引数:"):
            in_args = True
            last_param = None
            continue
        if in_args:
            if not s_stripped:
                # blank line ends args section
                break
            if lower.startswith(("returns:", "返り値:", "raises:", "例:")):
                break

            # Match "name: description" with optional leading bullets/indent
            m = re.match(r"^\s*(?:[-*]\s+)?([*]{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", s)
            if m:
                name = m.group(1)
                desc = _shorten(m.group(2))
                param_docs[name] = desc
                last_param = name
            else:
                # Continuation line for previous param
                if last_param is not None:
                    cont = _shorten(s_stripped)
                    if cont:
                        param_docs[last_param] = _shorten(param_docs[last_param] + " " + cont)

    # Normalize keys to match our stub parameter names (**_params vs **params)
    # We'll display docs for known parameters only; **params maps to **_params
    if "**params" in param_docs and "**_params" not in param_docs:
        param_docs["**_params"] = param_docs["**params"]

    return summary, param_docs


def _render_method_from_generate(shape_name: str, shape_cls: type) -> str:
    """Render a Protocol method from a shape class `generate()` signature.

    - Parameters become keyword-only (prefixed by `*`) to match runtime API.
    - Unknown/extra params are accepted via `**_params: Any` to avoid false negatives.
    - Adds inline comment lines with short argument descriptions under the stub.
      (keeps the `def ... -> Geometry: ...` on one line to satisfy tests/parsers)
    """
    # Default to flexible signature if anything goes wrong
    try:
        gen = getattr(shape_cls, "generate")
        sig = inspect.signature(gen)
        hints = {}
        try:
            hints = get_type_hints(gen, globalns=getattr(gen, "__globals__", {}))
        except Exception:
            hints = {}

        params_out: list[str] = []
        for p in sig.parameters.values():
            if p.name in ("self", "cls"):
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
                # Skip *args; we will add **_params anyway
                continue
            if p.kind in (inspect.Parameter.VAR_KEYWORD,):
                # Will explicitly add **_params below
                continue

            ann = hints.get(p.name, p.annotation)
            ann_s = _format_type(ann) if ann is not inspect._empty else "Any"
            # In stubs, prefer `= ...` for defaults to keep API stable
            default_s = " = ..." if p.default is not inspect._empty else ""
            params_out.append(f"{p.name}: {ann_s}{default_s}")

        # Build parameter list with keyword-only enforcement
        if params_out:
            paramlist = "*, " + ", ".join(params_out) + ", **_params: Any"
        else:
            paramlist = "**_params: Any"

        # Emit a block-style stub with a proper docstring, then ellipsis body
        lines: list[str] = []
        lines.append(f"    def {shape_name}(self, {paramlist}) -> Geometry:\n")

        summary, pdocs = _extract_param_docs(gen)
        # Compose docstring
        doc: list[str] = []
        if summary:
            doc.append(summary)
        if pdocs:
            if doc:
                doc.append("")
            doc.append("引数:")
            # Keep the order of parameters as declared
            for p in sig.parameters.values():
                if p.name in ("self", "cls"):
                    continue
                key = p.name
                if key == "params":
                    key = "**_params"
                desc = pdocs.get(key)
                if desc:
                    doc.append(f"    {key}: {desc}")

            if doc:
                # Write a triple-quoted docstring
                lines.append("        \"\"\"" + ("\n" if len(doc) > 1 else ""))
                for i, dl in enumerate(doc):
                    if dl:
                        lines.append(f"        {dl}\n")
                    else:
                        lines.append("\n")
                lines.append("        \"\"\"\n")
        # Ellipsis function body for stub
        lines.append("        ...\n")

        return "".join(lines)
    except Exception:
        # Fallback (generic)
        return f"    def {shape_name}(self, **_params: Any) -> Geometry: ...\n"


def _annotation_for_effect_param(tp: Any, imports: set[str]) -> str:
    """Best-effort annotation rendering for effect params with minimal imports.

    - Prefer PEP 604 (|) unions to avoid typing imports.
    - Map Tuple[float, float, float] to Vec3 and add `from common.types import Vec3`.
    - Map NoneType to `None`.
    - Fallback to `Any` when unsure.
    """
    try:
        if tp is inspect._empty:
            return "Any"
        origin = get_origin(tp)
        args = get_args(tp)

        # NoneType aliasing
        if tp is type(None):  # noqa: E721
            return "None"

        # Literal[...] -> degrade to base type (str/int/float) to avoid imports
        from typing import Literal  # type: ignore
        if origin is Literal:
            # Choose the first literal's Python type representation
            if args:
                lit = args[0]
                return _format_type(type(lit))
            return "Any"

        # Union -> use PEP 604 syntax
        from typing import Union  # type: ignore
        if origin is Union:
            parts = [
                _annotation_for_effect_param(a, imports)  # recurse for nested
                for a in args
            ]
            # De-duplicate and join
            uniq = []
            for p in parts:
                if p not in uniq:
                    uniq.append(p)
            return " | ".join(uniq)

        # Tuple[...] -> check Vec3 first, else use builtin tuple[...] syntax
        if origin in (tuple,):
            if len(args) == 3 and all(a is float for a in args):
                imports.add("from common.types import Vec3")
                return "Vec3"
            inner = ", ".join(_annotation_for_effect_param(a, imports) for a in args)
            return f"tuple[{inner}]"

        # Builtins and simple names
        s = _format_type(tp)
        # Normalize common noisy tokens
        s = s.replace("NoneType", "None")
        return s or "Any"
    except Exception:
        return "Any"


def _render_pipeline_protocol(effect_names: Iterable[str]) -> tuple[str, list[str]]:
    """Render `_PipelineBuilder` and `_Effects` protocols and return (text, extra_imports).

    Emits block-style method stubs with docstrings so IDEs can show tooltips.
    """
    extra_imports: set[str] = set()
    lines: list[str] = []

    # Builder protocol
    lines.append("class _PipelineBuilder(Protocol):\n")

    from effects.registry import get_effect
    for name in sorted(effect_names):
        try:
            fn = get_effect(name)
        except Exception:
            continue
        try:
            sig = inspect.signature(fn)
            hints: dict[str, Any] = {}
            try:
                hints = get_type_hints(fn, globalns=getattr(fn, "__globals__", {}))
            except Exception:
                hints = {}

            params_out: list[str] = []
            for p in sig.parameters.values():
                if p.name in ("g", "self", "cls"):
                    continue
                if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
                    continue
                if p.kind in (inspect.Parameter.VAR_KEYWORD,):
                    continue

                ann = hints.get(p.name, p.annotation)
                ann_s = _annotation_for_effect_param(ann, extra_imports)
                default_s = " = ..." if p.default is not inspect._empty else ""
                params_out.append(f"{p.name}: {ann_s}{default_s}")

            # Emit param meta comments (type/min/max/choices) before method
            meta = getattr(fn, "__param_meta__", None)
            if isinstance(meta, dict):
                for p in sig.parameters.values():
                    if p.name in ("g", "self", "cls"):
                        continue
                    rules = meta.get(p.name)
                    if not isinstance(rules, dict):
                        continue
                    parts: list[str] = []
                    if "type" in rules:
                        parts.append(f"type={rules['type']}")
                    has_min = "min" in rules
                    has_max = "max" in rules
                    if has_min and has_max:
                        parts.append(f"range=[{rules['min']}, {rules['max']}]")
                    elif has_min:
                        parts.append(f"min={rules['min']}")
                    elif has_max:
                        parts.append(f"max={rules['max']}")
                    if parts:
                        lines.append(f"    # meta: {p.name} (" + ", ".join(parts) + ")\n")
                    if "choices" in rules:
                        try:
                            seq = list(rules.get("choices"))
                        except Exception:
                            seq = []
                        if seq:
                            preview = ", ".join(map(repr, seq[:6])) + (" …" if len(seq) > 6 else "")
                            lines.append(f"    # choices: {p.name} in [{preview}]\n")

            if params_out:
                paramlist = "*, " + ", ".join(params_out) + ", **_params: Any"
            else:
                paramlist = "**_params: Any"

            # Begin block-style def to allow docstring body
            lines.append(f"    def {name}(self, {paramlist}) -> _PipelineBuilder:\n")

            # Compose docstring from function doc + param meta as fallback
            summary, pdocs = _extract_param_docs(fn)
            doc_lines: list[str] = []
            if summary:
                doc_lines.append(summary)

            # Build Args section using available docs or meta
            # Prepare a meta mapping for quick lookup
            meta = getattr(fn, "__param_meta__", None)
            meta_map = meta if isinstance(meta, dict) else {}

            # Create Args only if we have something to say
            arg_docs: list[str] = []
            for p in sig.parameters.values():
                if p.name in ("g", "self", "cls"):
                    continue
                key = p.name
                text = None
                if pdocs:
                    text = pdocs.get(key)
                if not text:
                    rules = meta_map.get(key) if isinstance(meta_map, dict) else None
                    if isinstance(rules, dict):
                        parts: list[str] = []
                        if "type" in rules:
                            parts.append(str(rules["type"]))
                        if "min" in rules or "max" in rules:
                            lo = rules.get("min")
                            hi = rules.get("max")
                            if lo is not None and hi is not None:
                                parts.append(f"range [{lo}, {hi}]")
                            elif lo is not None:
                                parts.append(f"min {lo}")
                            elif hi is not None:
                                parts.append(f"max {hi}")
                        ch = rules.get("choices") if isinstance(rules, dict) else None
                        if ch is not None:
                            try:
                                seq = list(ch)
                            except Exception:
                                seq = []
                            if seq:
                                preview = ", ".join(map(repr, seq[:6]))
                                parts.append(f"choices {{ {preview}{' …' if len(seq) > 6 else ''} }}")
                        if parts:
                            text = ", ".join(parts)
                if text:
                    arg_docs.append(f"    {key}: {text}")

            if arg_docs:
                if doc_lines:
                    doc_lines.append("")
                doc_lines.append("引数:")
                doc_lines.extend(arg_docs)

            if doc_lines:
                lines.append("        \"\"\"" + ("\n" if len(doc_lines) > 1 else ""))
                for dl in doc_lines:
                    if dl:
                        lines.append(f"        {dl}\n")
                    else:
                        lines.append("\n")
                lines.append("        \"\"\"\n")
            lines.append("        ...\n")
        except Exception:
            lines.append(f"    def {name}(self, **_params: Any) -> _PipelineBuilder: ...\n")

    # Common builder utilities
    lines.append("    def build(self) -> Pipeline: ...\n")
    lines.append("    def strict(self, enabled: bool = ...) -> _PipelineBuilder: ...\n")
    lines.append("    def cache(self, *, maxsize: int | None) -> _PipelineBuilder: ...\n")
    lines.append("    def __call__(self, g: Geometry) -> Geometry: ...\n\n")

    # Effects holder
    lines.append("class _Effects(Protocol):\n")
    lines.append("    @property\n")
    lines.append("    def pipeline(self) -> _PipelineBuilder: ...\n\n")

    return "".join(lines), sorted(extra_imports)


def _render_pyi(shape_names: Iterable[str]) -> str:
    header = (
        "# This file is auto-generated by scripts/gen_g_stubs.py. DO NOT EDIT.\n"
        "# Regenerate with: python -m scripts.gen_g_stubs\n\n"
    )

    lines: list[str] = [header]
    lines.append("from typing import Any, Protocol, TypedDict, TypeAlias\n")
    lines.append("from engine.core.geometry import Geometry as Geometry\n")
    lines.append("from api.pipeline import Pipeline as Pipeline\n\n")
    # ---- Shared spec/JSON types ----
    lines.append("JSONScalar: TypeAlias = int | float | str | bool | None\n")
    lines.append("JSONLike: TypeAlias = JSONScalar | list['JSONLike'] | dict[str, 'JSONLike']\n\n")
    lines.append("class PipelineSpecStep(TypedDict):\n")
    lines.append("    name: str\n")
    lines.append("    params: dict[str, JSONLike]\n\n")
    lines.append("PipelineSpec: TypeAlias = list[PipelineSpecStep]\n\n")

    lines.append("class _GShapes(Protocol):\n")
    # Import shapes registry locally to inspect signatures
    # Install minimal dummies for optional heavy deps to avoid ImportError during stub gen
    def _install_dummy_deps() -> None:
        # numba.njit used in some shapes; define as no-op decorator
        try:
            import numba  # noqa: F401
        except Exception:
            m = types.ModuleType("numba")
            def _njit(*_a, **_k):
                def deco(fn):
                    return fn
                return deco
            m.njit = _njit  # type: ignore[attr-defined]
            sys.modules["numba"] = m

        # fontTools used by text shape; create minimal stubs
        try:
            import fontTools  # noqa: F401
        except Exception:
            ft = types.ModuleType("fontTools")
            pens = types.ModuleType("fontTools.pens")
            rec = types.ModuleType("fontTools.pens.recordingPen")
            class RecordingPen:  # pragma: no cover - dummy
                def __init__(self, *a, **k):
                    pass
            rec.RecordingPen = RecordingPen
            ttLib = types.ModuleType("fontTools.ttLib")
            class TTFont:  # pragma: no cover - dummy
                def __init__(self, *a, **k):
                    pass
                def __getitem__(self, key):
                    return types.SimpleNamespace(unitsPerEm=1000)
            ttLib.TTFont = TTFont
            sys.modules["fontTools"] = ft
            sys.modules["fontTools.pens"] = pens
            sys.modules["fontTools.pens.recordingPen"] = rec
            sys.modules["fontTools.ttLib"] = ttLib

    _install_dummy_deps()
    import shapes  # noqa: F401
    from api.shape_registry import get_shape_generator
    for name in sorted(shape_names):
        try:
            shape_cls = get_shape_generator(name)
        except Exception:
            shape_cls = None  # type: ignore
        if shape_cls is None:
            lines.append(f"    def {name}(self, **_params: Any) -> Geometry: ...\n")
        else:
            lines.append(_render_method_from_generate(name, shape_cls))
    lines.append("\n")

    # Pipeline Protocols for effects
    from effects.registry import list_effects
    proto_body, extra_imports = _render_pipeline_protocol(list_effects())
    # Add any extra imports (e.g., Vec3) below the standard ones
    for imp in extra_imports:
        if imp.startswith("from common.types"):
            lines.append(imp + "\n")
    if extra_imports:
        lines.append("\n")
    lines.append(proto_body)

    # Re-exports to match runtime API surface
    lines.append("from .shape_factory import ShapeFactory as ShapeFactory\n")
    lines.append("\n")
    lines.append("G: _GShapes\n")
    lines.append("E: _Effects\n")
    lines.append("from .runner import run_sketch as run_sketch, run_sketch as run\n")
    # Precise function signatures for pipeline spec helpers
    lines.append("def to_spec(pipeline: Pipeline) -> PipelineSpec: ...\n")
    lines.append("def from_spec(spec: PipelineSpec) -> Pipeline: ...\n")
    lines.append("def validate_spec(spec: PipelineSpec) -> None: ...\n")
    lines.append("\n")
    # Keep exported names aligned with runtime __all__
    lines.append(
        "__all__ = [\n"
        "    'G', 'E', 'run_sketch', 'run', 'ShapeFactory', 'Geometry', 'to_spec', 'from_spec', 'validate_spec',\n"
        "]\n"
    )

    return "".join(lines)


def main() -> None:
    # Import inside main to avoid import-time side effects during tooling
    import shapes  # noqa: F401  # ensure registry side-effects
    from api.shape_registry import list_registered_shapes

    all_names = list_registered_shapes()
    valid = [n for n in all_names if _is_valid_identifier(n)]
    skipped = sorted(set(all_names) - set(valid))

    content = _render_pyi(valid)

    out_path = Path(__file__).resolve().parent.parent / "api" / "__init__.pyi"
    out_path.write_text(content, encoding="utf-8")

    notice = f"Wrote {out_path} (shapes={len(valid)}"
    if skipped:
        notice += f", skipped_invalid={skipped}"
    notice += ")"
    print(notice)


if __name__ == "__main__":
    main()
