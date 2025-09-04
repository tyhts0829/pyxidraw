# ADR 0007: ログ出力ポリシー

- ステータス: Accepted
- 日付: 2025-09-04

## 背景
print と logging が混在すると可観測性と運用性が落ちる。

## 決定
- Engine 層は `logging` を使用（INFO/DEBUG/ERROR 適切に）。
- CLI/ベンチは UX/仕様上の理由で `print` を許容。
- 外部依存のデバッグツール（例: icecream）は本体から排除。

## 影響
ログ収集/フィルタが統一でき、開発/運用の双方で扱いやすい。

## 代替案
全面 print は柔軟だが集約性が低い。全面 logging は CLI UX を損ねる。

