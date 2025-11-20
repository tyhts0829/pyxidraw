"""src 配下の Python モジュール行数を集計し、テキストで可視化するスクリプト。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModuleStat:
    """単一モジュールに対する行数統計。"""

    module: str
    lines: int


def count_lines(path: Path) -> int:
    """ファイルの行数を UTF-8 で数える。"""
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def collect_module_stats(src_root: Path) -> list[ModuleStat]:
    """src 直下の Python ファイルを列挙して行数統計を得る。"""
    stats: list[ModuleStat] = []
    for file_path in sorted(src_root.rglob("*.py")):
        if file_path.is_file():
            module_name = file_path.relative_to(src_root).as_posix()
            stats.append(ModuleStat(module=module_name, lines=count_lines(file_path)))
    return sorted(stats, key=lambda stat: stat.lines, reverse=True)


def render_table(stats: list[ModuleStat]) -> None:
    """集計結果を表形式と簡易バーで出力する。"""
    if not stats:
        print("Python ファイルが見つかりませんでした。")
        return

    max_module_len = max(len(stat.module) for stat in stats)
    max_lines = max(stat.lines for stat in stats)
    bar_width = 40

    header = f"{'module':<{max_module_len}} | lines | visualization"
    print(header)
    print("-" * len(header))
    for stat in stats:
        ratio = stat.lines / max_lines if max_lines else 0
        filled = int(ratio * bar_width)
        bar = "#" * filled
        print(f"{stat.module:<{max_module_len}} | {stat.lines:5d} | {bar}")

    total = sum(stat.lines for stat in stats)
    print("-" * len(header))
    print(f"{'total':<{max_module_len}} | {total:5d} |")


def main() -> None:
    """ルートから src を探索して結果を表示する。"""
    repo_root = Path(__file__).resolve().parent
    src_root = repo_root / "src"
    if not src_root.exists():
        raise SystemExit("src ディレクトリが見つかりません。")

    stats = collect_module_stats(src_root)
    render_table(stats)


if __name__ == "__main__":
    main()
