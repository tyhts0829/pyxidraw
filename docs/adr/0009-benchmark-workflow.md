# ADR 0009: ベンチマーク基準化ワークフロー

- ステータス: Accepted
- 日付: 2025-09-04

## 背景
変更の性能影響を継続的に把握するため、結果保存/比較の最小手順を公式化する。

## 決定
- baseline を保存し、current と比較する最小フローを README に掲載。
- `python -m benchmarks run -o benchmark_results/baseline`
- `python -m benchmarks run -o benchmark_results/current`
- `python -m benchmarks compare baseline.json current.json`

## 影響
性能回帰/改善の可視化が容易になる。

## 代替案
手動記録や ad-hoc スクリプト運用は共有性と再現性が低い。

