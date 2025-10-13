"""
どこで: `util.paths`。
何を: 画像/G-code 保存先ディレクトリの生成と解決ユーティリティを提供する。
なぜ: ランタイムから簡潔に保存先を扱え、並行呼び出しでも安全に作成できるようにするため。
"""

from __future__ import annotations

from pathlib import Path

from .utils import _find_project_root


def ensure_screenshots_dir() -> Path:
    """スクリーンショット出力先 `data/screenshot/` を作成して返す。

    - プロジェクトルート直下の `data/screenshot/` に作成する。
    - 既存の場合もそのまま Path を返す。
    - 並行呼び出しに対して `exist_ok=True` で安全。
    """
    root = _find_project_root(Path(__file__).parent)
    out = root / "data" / "screenshot"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_gcode_dir() -> Path:
    """G-code 出力先 `data/gcode/` を作成して返す。

    - プロジェクトルート直下に `data/gcode` を作成する。
    - 親 `data/` も同時に作成される。
    - 既存の場合もそのまま Path を返す。
    - 並行呼び出しに対して `exist_ok=True` で安全。
    """
    root = _find_project_root(Path(__file__).parent)
    out = root / "data" / "gcode"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_video_dir() -> Path:
    """動画出力先 `data/video/` を作成して返す。

    - プロジェクトルート直下に `data/video` を作成する。
    - 親 `data/` も同時に作成される。
    - 既存の場合もそのまま Path を返す。
    - 並行呼び出しに対して `exist_ok=True` で安全。
    """
    root = _find_project_root(Path(__file__).parent)
    out = root / "data" / "video"
    out.mkdir(parents=True, exist_ok=True)
    return out
