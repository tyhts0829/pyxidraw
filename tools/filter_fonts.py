from __future__ import annotations

"""
どこで: tools.filter_fonts。
何を: 指定ディレクトリから基本的な英字グリフを含むフォントだけを抽出し、JSON に書き出す。
なぜ: Parameter GUI のフォント候補を絞り、利用できないフォントをプルダウンから除外するため。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from util.fonts import glob_font_files, os_font_dirs, resolve_search_dirs  # type: ignore
from util.utils import _find_project_root, load_config  # type: ignore

UPPERCASE = {ord(ch) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
LOWERCASE = {ord(ch) for ch in "abcdefghijklmnopqrstuvwxyz"}
DEFAULT_EXTS = (".ttf", ".otf", ".ttc")


def _project_root() -> Path:
    return _find_project_root(Path(__file__).parent)


def _default_output_path() -> Path:
    return _project_root() / "data" / "fonts" / "usable_fonts.json"


def _config_search_dirs() -> tuple[list[Path], bool]:
    cfg = load_config() or {}
    fonts_cfg = cfg.get("fonts", {}) if isinstance(cfg, dict) else {}
    raw_dirs = fonts_cfg.get("search_dirs", []) if isinstance(fonts_cfg, dict) else []
    if isinstance(raw_dirs, (str, Path)):
        raw_dirs = [str(raw_dirs)]
    try:
        dirs = resolve_search_dirs(raw_dirs)
    except Exception:
        dirs = []
    include_os = True
    try:
        include_os = bool(fonts_cfg.get("include_os", True))
    except Exception:
        include_os = True
    return dirs, include_os


def _iter_faces(path: Path) -> Iterable[tuple[int, Any]]:
    try:
        from fontTools.ttLib import TTCollection, TTFont  # type: ignore
    except Exception as exc:  # pragma: no cover - fontTools は任意依存
        raise SystemExit(
            "fontTools が必要です。`pip install fonttools` を実行してください。"
        ) from exc

    if path.suffix.lower() == ".ttc":
        coll: Any | None = None
        try:
            coll = TTCollection(path)
            for idx, font in enumerate(coll.fonts):
                yield idx, font
        except Exception:
            return
        finally:
            try:
                if coll is not None:
                    coll.close()
            except Exception:
                pass
    else:
        font: Any | None = None
        try:
            font = TTFont(path)
            yield 0, font
        except Exception:
            return
        finally:
            try:
                if font is not None:
                    font.close()
            except Exception:
                pass


def _has_basic_letters(tt_font: Any) -> tuple[bool, bool]:
    try:
        cmap = tt_font.getBestCmap()
    except Exception:
        return False, False
    if not cmap:
        return False, False
    codes = set(cmap.keys())
    has_upper = UPPERCASE.issubset(codes)
    has_lower = LOWERCASE.issubset(codes)
    return has_upper, has_lower


def _get_name_record(tt_font: Any, name_id: int) -> str | None:
    try:
        name_table = tt_font["name"]  # type: ignore[index]
    except Exception:
        return None
    for platform_id, enc_id in ((3, 1), (1, 0), (0, 1)):
        try:
            rec = name_table.getName(name_id, platform_id, enc_id)
        except Exception:
            rec = None
        if rec is None:
            continue
        try:
            return rec.toUnicode()  # type: ignore[call-arg]
        except Exception:
            try:
                return str(rec)
            except Exception:
                continue
    return None


def _display_name(tt_font: Any, path: Path, font_index: int) -> str:
    family = _get_name_record(tt_font, 1) or path.stem
    subfamily = _get_name_record(tt_font, 2)
    base = f"{family} {subfamily}".strip() if subfamily else family
    if path.suffix.lower() == ".ttc":
        return f"{base} (idx {font_index})"
    return base


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _collect_files(
    search_dirs: list[Path], include_os: bool, exts: tuple[str, ...]
) -> tuple[list[Path], list[Path]]:
    os_dirs = os_font_dirs() if include_os else []
    dirs = list(search_dirs) + os_dirs
    files = glob_font_files(dirs, exts)
    return files, os_dirs


def _filter_fonts(files: Iterable[Path], os_dirs: list[Path]) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for path in files:
        source = "os" if any(_is_under(path, d) for d in os_dirs) else "config"
        for face_idx, font in _iter_faces(path):
            has_upper, has_lower = _has_basic_letters(font)
            if not (has_upper or has_lower):
                continue
            entry = {
                "path": str(path.resolve()),
                "font_index": int(face_idx),
                "display_name": _display_name(font, path, face_idx),
                "has_uppercase": has_upper,
                "has_lowercase": has_lower,
                "source": source,
            }
            usable.append(entry)
            # 最初に見つかった usable face を採用（path ごとに 1 件）
            break
    usable.sort(key=lambda e: (str(e.get("display_name", "")).lower(), e["path"]))
    return usable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基本英字グリフを持つフォントだけを抽出して JSON に書き出す。"
    )
    parser.add_argument(
        "-d",
        "--search-dir",
        action="append",
        default=[],
        help="追加の検索ディレクトリ。複数指定可。",
    )
    parser.add_argument(
        "--include-os",
        dest="include_os",
        action="store_true",
        default=None,
        help="OS フォントを含める（既定）。",
    )
    parser.add_argument(
        "--no-include-os",
        dest="include_os",
        action="store_false",
        help="OS フォントを含めない。",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="出力先 JSON パス。未指定時は data/fonts/usable_fonts.json。",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="探索する拡張子を追加で指定（既定: .ttf/.otf/.ttc）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="書き込みせずに結果サマリを表示する。",
    )
    return parser.parse_args()


def main(argv: list[str]) -> int:
    args = _parse_args()
    cfg_dirs, cfg_include_os = _config_search_dirs()
    add_dirs_raw = args.search_dir if isinstance(args.search_dir, list) else []
    add_dirs: list[Path] = []
    for raw in add_dirs_raw:
        try:
            p = Path(raw).expanduser()
            add_dirs.append(p if p.is_absolute() else (_project_root() / p).resolve())
        except Exception:
            continue

    search_dirs = cfg_dirs + add_dirs
    include_os = cfg_include_os if args.include_os is None else bool(args.include_os)
    exts = tuple(args.ext) if args.ext else DEFAULT_EXTS
    files, os_dirs = _collect_files(search_dirs, include_os, exts)

    usable = _filter_fonts(files, os_dirs)
    payload = {
        "fonts": usable,
        "required": {
            "needs_upper_or_lower": True,
            "uppercase": [chr(c) for c in sorted(UPPERCASE)],
            "lowercase": [chr(c) for c in sorted(LOWERCASE)],
        },
    }

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    output = args.output if args.output is not None else _default_output_path()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - IO 失敗は CLI エラーとして扱う
        sys.stderr.write(f"書き込みに失敗しました: {exc}\n")
        return 1

    print(f"{len(usable)} 個のフォントを書き出しました: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI エントリ
    raise SystemExit(main(sys.argv[1:]))
