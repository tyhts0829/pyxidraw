# ベンチマーク並列実行ガイド

目的: 実運用で安定かつ効率的に `python -m benchmarks run` を並列実行するための実践ガイド。

- 推奨ワーカー数: `論理CPU数 × 0.8` を目安（例: 8C/16T → `--workers 13`）。
- CPU バウンド主体: `--parallel` を有効にし、他の重いジョブと同居させない。
- I/O バウンド・メモリ圧: `--workers` を半減（例: `--workers 6`）し、ノイズを低減。
- 温度/スロットリング: 長時間の連続実行時は `--warmup 1 --runs 10` に調整して熱の飽和を避ける。
- 固定化: 同一マシン・同一電源モードで比較。必要に応じて `taskset` 等で CPU ピニング。
- スキップ/再実行: 失敗時は `benchmark_results/failed_targets.json` を `--from-file` に渡すと再測定できる。

例:

```
python -m benchmarks run --parallel --workers 12
python -m benchmarks run --from-file benchmark_results/failed_targets.json --parallel
```

