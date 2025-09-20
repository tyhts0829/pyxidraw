"""
pytest-benchmark の JSON 出力（--benchmark-json=perf.json）を要約して
Markdown/CSV/JSON を生成する軽量スクリプト。

使い方:
    python -m tools.bench.summarize perf.json \
        --out-md benchmark_results/summary.md \
        --out-csv benchmark_results/summary.csv \
        --out-json benchmark_results/summary.json

出力列（行単位）:
- name: pytest テスト名
- case: benchmark.extra_info の case（なければ name）
- N/M: extra_info に含まれる頂点数/ライン数（なければ空）
- rss_delta / alloc_peak: メモリ指標（extra_info）
- min/median/mean/stddev/ops/rounds/iterations: 統計（ops は mean の逆数から計算）
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional


@dataclass
class Row:
    name: str
    case: str
    N: Optional[int]
    M: Optional[int]
    min: float
    median: float
    mean: float
    stddev: float
    ops: float
    rounds: int
    iterations: Optional[int]
    rss_delta: Optional[int]
    alloc_peak: Optional[int]


def _get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_rows(data: Mapping[str, Any]) -> List[Row]:
    benches = data.get("benchmarks", [])
    rows: List[Row] = []
    for b in benches:
        name = str(b.get("name", ""))
        extra = b.get("extra_info", {}) or {}
        stats = b.get("stats", {}) or {}
        case = str(extra.get("case") or name)
        N = extra.get("N")
        M = extra.get("M")
        # 基本統計
        min_v = float(stats.get("min", 0.0))
        median_v = float(stats.get("median", 0.0))
        mean_v = float(stats.get("mean", 0.0))
        std_v = float(stats.get("stddev", 0.0))
        rounds = int(stats.get("rounds", 0))
        iterations = stats.get("iterations")
        # ops は mean の逆数（mean==0 は 0）
        ops = 1.0 / mean_v if mean_v else 0.0
        rss_delta = extra.get("rss_delta")
        alloc_peak = extra.get("alloc_peak")
        rows.append(
            Row(
                name=name,
                case=case,
                N=int(N) if isinstance(N, (int, float)) else None,
                M=int(M) if isinstance(M, (int, float)) else None,
                min=min_v,
                median=median_v,
                mean=mean_v,
                stddev=std_v,
                ops=ops,
                rounds=rounds,
                iterations=int(iterations) if isinstance(iterations, (int, float)) else None,
                rss_delta=int(rss_delta) if isinstance(rss_delta, (int, float)) else None,
                alloc_peak=int(alloc_peak) if isinstance(alloc_peak, (int, float)) else None,
            )
        )

    # 表示安定のため case -> name の順でソート
    rows.sort(key=lambda r: (r.case, r.name))
    return rows


def write_md(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(
            "| case | name | N | M | mean (s) | median (s) | stddev | ops | rounds | rss_delta | alloc_peak |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {case} | {name} | {N} | {M} | {mean:.6e} | {median:.6e} | {std:.3e} | {ops:.2f} | {rounds} | {rss} | {alloc} |\n".format(
                    case=r.case,
                    name=r.name,
                    N=r.N if r.N is not None else "",
                    M=r.M if r.M is not None else "",
                    mean=r.mean,
                    median=r.median,
                    std=r.stddev,
                    ops=r.ops,
                    rounds=r.rounds,
                    rss=r.rss_delta if r.rss_delta is not None else "",
                    alloc=r.alloc_peak if r.alloc_peak is not None else "",
                )
            )


def write_csv(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "name",
        "N",
        "M",
        "min",
        "median",
        "mean",
        "stddev",
        "ops",
        "rounds",
        "iterations",
        "rss_delta",
        "alloc_peak",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "case": r.case,
                    "name": r.name,
                    "N": r.N if r.N is not None else "",
                    "M": r.M if r.M is not None else "",
                    "min": f"{r.min:.9e}",
                    "median": f"{r.median:.9e}",
                    "mean": f"{r.mean:.9e}",
                    "stddev": f"{r.stddev:.9e}",
                    "ops": f"{r.ops:.3f}",
                    "rounds": r.rounds,
                    "iterations": r.iterations if r.iterations is not None else "",
                    "rss_delta": r.rss_delta if r.rss_delta is not None else "",
                    "alloc_peak": r.alloc_peak if r.alloc_peak is not None else "",
                }
            )


def write_json(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize pytest-benchmark JSON to MD/CSV/JSON")
    parser.add_argument("input", help="Path to pytest-benchmark JSON (perf.json)")
    parser.add_argument("--out-md", dest="out_md", help="Output Markdown path")
    parser.add_argument("--out-csv", dest="out_csv", help="Output CSV path")
    parser.add_argument("--out-json", dest="out_json", help="Output JSON path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = load_rows(data)

    if args.out_md:
        write_md(Path(args.out_md), rows)
    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    if args.out_json:
        write_json(Path(args.out_json), rows)

    # 出力先未指定の場合は簡易プレビューを標準出力へ
    if not (args.out_md or args.out_csv or args.out_json):
        for r in rows:
            print(f"{r.case:32s} mean={r.mean:.6e} ops={r.ops:.2f} rounds={r.rounds}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI エントリ
    raise SystemExit(main())
