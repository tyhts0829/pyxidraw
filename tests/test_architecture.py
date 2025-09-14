"""
アーキテクチャテスト

目的:
- 数値レイヤ（外側→内側のみ許可、同層は許可）
- 個別禁止エッジ（architecture.md の契約）

注:
- `src/scripts/` は対象外（開発補助のための例外を許容）。
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# レイヤ定義（数値が小さいほど内側）
# L0: Core/Base, L1: Domain/Transforms, L2: Infra/Runtime, L3: API/Entry
LAYER_MAP = {
    ("common",): 0,
    ("util",): 0,
    ("engine", "core"): 0,
    ("effects",): 1,
    ("shapes",): 1,
    ("engine", "render"): 2,
    ("engine", "pipeline"): 2,
    ("engine", "ui"): 2,
    ("engine", "io"): 2,
    ("engine", "monitor"): 2,
    ("api",): 3,
}

CHECK_ROOTS = {"api", "engine", "effects", "shapes", "common", "util"}
ENGINE_FORBIDDEN_TARGET_SUBS = {"render", "pipeline", "ui", "io", "monitor"}


def iter_py_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for p in root.rglob("*.py"):
        # 除外: テスト自身、キャッシュ、開発補助スクリプト
        if any(part in {"tests", "__pycache__", "scripts"} for part in p.parts):
            continue
        yield p


def module_name_from_path(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC_DIR).with_suffix("")
    return ".".join(rel.parts)


def split_head(module: str) -> tuple[str, Optional[str]]:
    parts = module.split(".") if module else []
    if not parts:
        return "", None
    head = parts[0]
    second = parts[1] if len(parts) > 1 else None
    return head, second


def layer_of(module: str) -> Optional[int]:
    head, second = split_head(module)
    key: tuple[str, ...]
    if head == "engine" and second:
        key = (head, second)
    elif head:
        key = (head,)
    else:
        return None
    return LAYER_MAP.get(key)


def engine_subsection(module: str) -> Optional[str]:
    head, second = split_head(module)
    if head != "engine" or second is None:
        return None
    return second


def iter_import_edges(py_path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    src_mod = module_name_from_path(py_path)
    text = py_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield src_mod, alias.name  # 例: "numpy"
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # 相対 import
                src_parts = src_mod.split(".")
                base = ".".join(src_parts[: -node.level])
                mod = node.module or ""
                tgt = f"{base}.{mod}" if mod else base
                yield src_mod, tgt
            else:
                yield src_mod, node.module or ""


def is_forbidden_edge(src: str, tgt: str) -> bool:
    s_head, _ = split_head(src)
    t_head, _ = split_head(tgt)
    # 1) engine -> api を禁止
    if s_head == "engine" and t_head == "api":
        return True
    # 2) engine -> effects/shapes を禁止
    if s_head == "engine" and t_head in {"effects", "shapes"}:
        return True
    # 3) effects/shapes -> engine.{render,pipeline,ui,io,monitor} を禁止（engine.core への依存は可）
    if s_head in {"effects", "shapes"} and t_head == "engine":
        t_sec = engine_subsection(tgt)
        if t_sec in ENGINE_FORBIDDEN_TARGET_SUBS:
            return True
    # 4) registries 参照は api/effects/shapes 以外から禁止（engine/common/util からは不可）
    if tgt in {"effects.registry", "shapes.registry"}:
        if s_head not in {"api", "effects", "shapes"}:  # 例: engine/*, common/*, util/* から禁止
            return True
    return False


def within_check_scope(module: str) -> bool:
    head, _ = split_head(module)
    return head in CHECK_ROOTS


def collect_graph_and_violations() -> tuple[Dict[str, Set[str]], list[str], list[str]]:
    layering: list[str] = []
    forbidden: list[str] = []
    graph: Dict[str, Set[str]] = {}
    module_files: Dict[str, pathlib.Path] = {}

    # 1) すべてのモジュール名（ノード）を集める
    py_files: List[pathlib.Path] = list(iter_py_files(SRC_DIR))
    for py in py_files:
        mod = module_name_from_path(py)
        module_files[mod] = py
        graph.setdefault(mod, set())

    # 2) エッジを収集し、違反チェック
    for py in py_files:
        for src, tgt in iter_import_edges(py):
            if not within_check_scope(src) or not within_check_scope(tgt):
                continue
            s_layer = layer_of(src)
            t_layer = layer_of(tgt)
            if s_layer is not None and t_layer is not None and s_layer < t_layer:
                layering.append(f"[{py}] {src} (L{s_layer}) -> {tgt} (L{t_layer})")
            if is_forbidden_edge(src, tgt):
                forbidden.append(f"[{py}] {src} -> {tgt}")
            # グラフに追加（ターゲットが既知モジュールのときのみ）
            if src in graph and tgt in graph:
                graph[src].add(tgt)

    return graph, layering, forbidden


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    cycles: List[List[str]] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {u: WHITE for u in graph}
    stack: List[str] = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in graph.get(u, ()):  # 隣接
            if color[v] == WHITE:
                dfs(v)
            elif color[v] == GRAY:
                # サイクル検出（スタック上の v から末尾まで）
                try:
                    idx = stack.index(v)
                except ValueError:  # 安全側
                    idx = 0
                cycle = stack[idx:] + [v]
                # 正規化して重複削減（回転同値を除く）
                # 代表は辞書順最小の開始点に回転
                base = min(range(len(cycle) - 1), key=lambda i: cycle[i])
                norm = cycle[base:-1] + cycle[:base] + [cycle[base]]
                if norm not in cycles:
                    cycles.append(norm)
        stack.pop()
        color[u] = BLACK

    for node in list(graph):
        if color[node] == WHITE:
            dfs(node)
    return cycles


@pytest.mark.smoke
def test_architecture_import_rules():
    graph, layering, forbidden = collect_graph_and_violations()
    cycles = find_cycles(graph)
    msgs: list[str] = []
    if layering:
        msgs.append("Layer violations:\n" + "\n".join(layering))
    if forbidden:
        msgs.append("Forbidden-edge violations:\n" + "\n".join(forbidden))
    if cycles:
        # 表示数を制限してノイズを抑制
        rendered = [" -> ".join(c) for c in cycles[:10]]
        suffix = "\n(and more ...)" if len(cycles) > 10 else ""
        msgs.append("Import cycles detected (module-level):\n" + "\n".join(rendered) + suffix)
    if msgs:
        raise AssertionError("\n\n".join(msgs))
