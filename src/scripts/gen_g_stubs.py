"""
スタブ自動生成スクリプト（IDE 補完/型検査向け）

本モジュールは、公開 API の型スタブ `src/api/__init__.pyi` を自動生成する開発用ユーティリティである。
`G` 上に全 Shape 名を生やし、`E.pipeline` のビルダ API を `Protocol` として明示することで、
IDE の補完品質と mypy 等の型検査の精度を高める。

概要:
- 形状: `api.shape_registry.list_registered_shapes()` から登録済み Shape 名を収集し、
  `class _GShapes(Protocol)` に各 Shape 名のメソッドを生成する（戻り値は `Geometry`）。
- エフェクト/パイプライン: `effects.registry.list_effects()` を参照し、
  `class _PipelineBuilder(Protocol)` と `class _Effects(Protocol)` を生成する。
- 生成物: `G: _GShapes`, `E: _Effects`、およびパイプライン仕様（`PipelineSpec` 系）を再エクスポート。

設計方針:
- 依存の最小化: 生成時に重い依存（numba/fontTools/shapely など）を必要としないよう、
  `scripts.dummy_deps.install()` で軽量ダミーを注入して ImportError を回避する。
- 安定した署名: 取得したシグネチャから型注釈を最善で復元しつつ、既定値は `= ...` で表現して
  API 面を安定化する。解決できない場合はフォールバックして柔軟な `**_params: Any` を許容する。
- 文字列表現: typing 記法のノイズを極力取り除き、Union は PEP 604 形式（`A | B`）を優先。
  `tuple[float, float, float]` は `common.types.Vec3` へ正規化する。

CI/テスト連携:
- `tests/stubs/test_g_stub_sync.py` および `tests/stubs/test_pipeline_stub_sync.py` により、
  生成結果文字列（`generate_stubs_str()`）とディスク上の `api/__init__.pyi` の一致が検証される。
- 本スクリプトのコメントや docstring を日本語化しても、生成されるスタブ文字列の内容を
  変更しない限りテストには影響しない（機能は不変）。

使い方:
    python -m scripts.gen_g_stubs

補足:
- CI/ローカル双方で動作するよう、極めて防御的に introspection を行う。
- 例外時はフェイルクローズではなく、ジェネリックな署名を出力して IDE 補完を維持する。
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Iterable, get_args, get_origin, get_type_hints

from scripts.dummy_deps import install as install_dummy_deps


def _is_valid_identifier(name: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None


def _format_type(tp: Any) -> str:
    try:
        s = str(tp)
    except Exception:
        return "Any"
    # よくある typing の文字列表現を正規化
    s = s.replace("typing.", "")
    # 組み込み型の表示 <class 'int'> -> int へ簡素化
    s = re.sub(r"^<class '([a-zA-Z_][a-zA-Z0-9_\.]*)'>$", r"\1", s)
    return s


def _extract_param_docs(gen_obj: Any) -> tuple[str | None, dict[str, str]]:
    """`generate` の docstring から要約と引数説明を抽出する。

    戻り値は `(summary, {param: short_desc})`。
    極力寛容かつ防御的にパースする:
    - 英語の "Args:" と日本語の "引数:" の両方に対応。
    - 続行行は直前のパラメータの説明に連結する。
    - 各説明は 1 行の簡潔な文に短縮する（文末句読点は省略）。
    """
    doc = inspect.getdoc(gen_obj) or ""
    if not doc:
        return None, {}

    lines = doc.splitlines()
    # 要約: 最初の非空行（セクション見出しを除く）
    summary: str | None = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith(("args:", "returns:", "raises:", "引数:", "返り値:", "例:")):
            break
        summary = s
        break

    # Args ブロックを抽出
    in_args = False
    last_param: str | None = None
    param_docs: dict[str, str] = {}

    def _shorten(text: str, limit: int = 120) -> str:
        t = " ".join(text.split())  # 空白を正規化
        # 0.5 などの小数を壊さないよう英文ピリオドは温存。
        # 和文の句点は最初の出現で切って簡潔にする。
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
                # 空行で Args セクション終了
                break
            if lower.startswith(("returns:", "返り値:", "raises:", "例:")):
                break

            # 先頭の箇条書き/インデントは許容しつつ、"name: description" を抽出
            m = re.match(r"^\s*(?:[-*]\s+)?([*]{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", s)
            if m:
                name = m.group(1)
                desc = _shorten(m.group(2))
                param_docs[name] = desc
                last_param = name
            else:
                # 直前パラメータの説明の続行行
                if last_param is not None:
                    cont = _shorten(s_stripped)
                    if cont:
                        param_docs[last_param] = _shorten(param_docs[last_param] + " " + cont)

    # スタブ側の仮引数名に合わせてキーを正規化（**params -> **_params）
    # 既知のパラメータのみ表示する。
    if "**params" in param_docs and "**_params" not in param_docs:
        param_docs["**_params"] = param_docs["**params"]

    return summary, param_docs


def _render_method_from_generate(shape_name: str, shape_cls: type) -> str:
    """Shape クラスの `generate()` シグネチャから Protocol メソッドを生成する。

    - 実行時 API に合わせ、パラメータはキーワード専用（先頭に `*` 付与）。
    - 未知/追加の引数は `**_params: Any` で受け入れ、偽陰性を避ける。
    - スタブ直下に引数説明の簡易コメントを生成する（ただし `def ... -> Geometry: ...` は
      一行で維持し、テスト/パーサ互換性を確保）。
    """
    # 何か問題が起きた場合は柔軟な署名にフォールバック
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
                # `*args` はスキップ（後で `**_params` を必ず追加するため）
                continue
            if p.kind in (inspect.Parameter.VAR_KEYWORD,):
                # `**kwargs` は明示的に `**_params` を後段で追加
                continue

            ann = hints.get(p.name, p.annotation)
            ann_s = _format_type(ann) if ann is not inspect._empty else "Any"
            # 既定値はスタブ上では `= ...` を用いて API 面を安定化
            default_s = " = ..." if p.default is not inspect._empty else ""
            params_out.append(f"{p.name}: {ann_s}{default_s}")

        # キーワード専用の仮引数リストを構築
        if params_out:
            paramlist = "*, " + ", ".join(params_out) + ", **_params: Any"
        else:
            paramlist = "**_params: Any"

        # ブロック形式の定義を出力し、適切な docstring の後に省略記号本文を置く
        lines: list[str] = []
        lines.append(f"    def {shape_name}(self, {paramlist}) -> Geometry:\n")

        summary, pdocs = _extract_param_docs(gen)
        # docstring を組み立て
        doc: list[str] = []
        if summary:
            doc.append(summary)
        if pdocs:
            if doc:
                doc.append("")
            doc.append("引数:")
            # 宣言順を維持して引数を列挙
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
                # 三重引用符の docstring を書き出す
                lines.append('        """' + ("\n" if len(doc) > 1 else ""))
                for i, dl in enumerate(doc):
                    if dl:
                        lines.append(f"        {dl}\n")
                    else:
                        lines.append("\n")
                lines.append('        """\n')
        # スタブ本文は省略記号
        lines.append("        ...\n")

        return "".join(lines)
    except Exception:
        # フォールバック（汎用）
        return f"    def {shape_name}(self, **_params: Any) -> Geometry: ...\n"


def _annotation_for_effect_param(tp: Any, imports: set[str]) -> str:
    """エフェクト引数型の最善描画（追加 import を最小化）。

    - Union は typing を増やさないため PEP 604 記法（`|`）を優先。
    - `Tuple[float, float, float]` は `Vec3` に写像し、`from common.types import Vec3` を追補。
    - `NoneType` は `None` に正規化。
    - 不明な場合は `Any` にフォールバック。
    """
    try:
        if tp is inspect._empty:
            return "Any"
        origin = get_origin(tp)
        args = get_args(tp)

        # NoneType の別名化
        if tp is type(None):  # noqa: E721
            return "None"

        # Literal[...] -> 追加 import 回避のため基底型（str/int/float）に劣化
        from typing import Literal  # type: ignore

        if origin is Literal:
            # 最初のリテラルの Python 型表現を採用
            if args:
                lit = args[0]
                return _format_type(type(lit))
            return "Any"

        # Union -> PEP 604 記法に展開
        from typing import Union  # type: ignore

        if origin is Union:
            parts = [_annotation_for_effect_param(a, imports) for a in args]  # 再帰で解決
            # 重複を除いて連結
            uniq = []
            for p in parts:
                if p not in uniq:
                    uniq.append(p)
            return " | ".join(uniq)

        # Tuple[...] -> まず Vec3 判定、それ以外は組み込みの tuple[...] 記法
        if origin in (tuple,):
            if len(args) == 3 and all(a is float for a in args):
                imports.add("from common.types import Vec3")
                return "Vec3"
            inner = ", ".join(_annotation_for_effect_param(a, imports) for a in args)
            return f"tuple[{inner}]"

        # 組み込み/単純名
        s = _format_type(tp)
        # よくあるノイズの正規化
        s = s.replace("NoneType", "None")
        return s or "Any"
    except Exception:
        return "Any"


def _render_pipeline_protocol(effect_names: Iterable[str]) -> tuple[str, list[str]]:
    """`_PipelineBuilder` と `_Effects` の Protocol 本体を生成して返す。

    ブロック形式のメソッドスタブに docstring を付与し、IDE のツールチップ表示に活用できるようにする。
    戻り値は `(text, extra_imports)`（追加 import は主に `Vec3`）。
    """
    extra_imports: set[str] = set()
    lines: list[str] = []

    # ビルダ Protocol
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

            # メソッドの前にパラメータのメタ情報（type/min/max/choices）をコメント出力
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
                        ch = rules.get("choices")
                        seq: list[Any] = []
                        try:
                            if isinstance(ch, IterableABC):
                                seq = list(ch)  # 型検査用に Iterable を確認してから list 化
                        except Exception:
                            seq = []
                        if seq:
                            preview = ", ".join(map(repr, seq[:6])) + (" …" if len(seq) > 6 else "")
                            lines.append(f"    # choices: {p.name} in [{preview}]\n")

            if params_out:
                paramlist = "*, " + ", ".join(params_out) + ", **_params: Any"
            else:
                paramlist = "**_params: Any"

            # docstring 本文を持てるようブロック形式で定義開始
            lines.append(f"    def {name}(self, {paramlist}) -> _PipelineBuilder:\n")

            # 関数の docstring と param meta を併用して docstring を構築
            summary, pdocs = _extract_param_docs(fn)
            doc_lines: list[str] = []
            if summary:
                doc_lines.append(summary)

            # 利用可能な docs または meta から Args 節を構築
            # 参照しやすいよう meta の辞書を準備
            meta = getattr(fn, "__param_meta__", None)
            meta_map = meta if isinstance(meta, dict) else {}

            # 説明がある場合のみ Args を作成
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
                        seq: list[Any] = []
                        try:
                            if isinstance(ch, IterableABC):
                                seq = list(ch)  # 型検査用に Iterable を確認
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
                lines.append('        """' + ("\n" if len(doc_lines) > 1 else ""))
                for dl in doc_lines:
                    if dl:
                        lines.append(f"        {dl}\n")
                    else:
                        lines.append("\n")
                lines.append('        """\n')
            lines.append("        ...\n")
        except Exception:
            lines.append(f"    def {name}(self, **_params: Any) -> _PipelineBuilder: ...\n")

    # ビルダの共通ユーティリティ
    lines.append("    def build(self) -> Pipeline: ...\n")
    lines.append("    def strict(self, enabled: bool = ...) -> _PipelineBuilder: ...\n")
    lines.append("    def cache(self, *, maxsize: int | None) -> _PipelineBuilder: ...\n")
    lines.append("    def __call__(self, g: Geometry) -> Geometry: ...\n\n")

    # Effects ホルダー
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
    # ---- 共有の Spec/JSON 型 ----
    lines.append("JSONScalar: TypeAlias = int | float | str | bool | None\n")
    lines.append("JSONLike: TypeAlias = JSONScalar | list['JSONLike'] | dict[str, 'JSONLike']\n\n")
    lines.append("class PipelineSpecStep(TypedDict):\n")
    lines.append("    name: str\n")
    lines.append("    params: dict[str, JSONLike]\n\n")
    lines.append("PipelineSpec: TypeAlias = list[PipelineSpecStep]\n\n")

    lines.append("class _GShapes(Protocol):\n")
    # 署名を検査するためにローカルで shapes レジストリを import
    # 生成時の ImportError を避けるため、重い依存の最小ダミーを注入
    install_dummy_deps()
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

    # エフェクト用 Pipeline Protocol 群
    from effects.registry import list_effects

    proto_body, extra_imports = _render_pipeline_protocol(list_effects())
    # 追加 import（Vec3 等）があれば標準 import の下に追加
    for imp in extra_imports:
        if imp.startswith("from common.types"):
            lines.append(imp + "\n")
    if extra_imports:
        lines.append("\n")
    lines.append(proto_body)

    # 実行時 API の表面と整合する再エクスポート
    lines.append("from .shape_factory import ShapeFactory as ShapeFactory\n")
    lines.append("\n")
    lines.append("G: _GShapes\n")
    lines.append("E: _Effects\n")
    lines.append("from .runner import run_sketch as run_sketch, run_sketch as run\n")
    # Pipeline Spec ヘルパ関数の厳密なシグネチャ
    lines.append("def to_spec(pipeline: Pipeline) -> PipelineSpec: ...\n")
    lines.append("def from_spec(spec: PipelineSpec) -> Pipeline: ...\n")
    lines.append("def validate_spec(spec: PipelineSpec) -> None: ...\n")
    lines.append("\n")
    # 実行時の `__all__` と整合させる
    lines.append(
        "__all__ = [\n"
        "    'G', 'E', 'run_sketch', 'run', 'ShapeFactory', 'Geometry', 'to_spec', 'from_spec', 'validate_spec',\n"
        "]\n"
    )

    return "".join(lines)


def generate_stubs_str() -> str:
    """`api/__init__.pyi` の生成結果を文字列として返す。

    形状/エフェクトのレジストリが最小ダミー依存を伴って初期化されることを保証し、
    依存の薄い CI/テスト環境でも動作するようにする。
    """
    install_dummy_deps()
    # 副作用的な登録を確実に行う
    import effects  # noqa: F401
    import shapes  # noqa: F401
    from api.shape_registry import list_registered_shapes

    all_names = list_registered_shapes()
    valid = [n for n in all_names if _is_valid_identifier(n)]
    return _render_pyi(valid)


def main() -> None:
    # テストと同じパス解決で一貫した挙動を担保
    content = generate_stubs_str()
    # ログ表示用にスキップ一覧を再計算
    from api.shape_registry import list_registered_shapes

    all_names = list_registered_shapes()
    valid = [n for n in all_names if _is_valid_identifier(n)]
    skipped = sorted(set(all_names) - set(valid))

    out_path = Path(__file__).resolve().parent.parent / "api" / "__init__.pyi"
    out_path.write_text(content, encoding="utf-8")

    notice = f"Wrote {out_path} (shapes={len(valid)}"
    if skipped:
        notice += f", skipped_invalid={skipped}"
    notice += ")"
    print(notice)


if __name__ == "__main__":
    main()
