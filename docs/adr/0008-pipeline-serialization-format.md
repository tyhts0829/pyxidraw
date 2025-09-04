# ADR 0008: パイプラインのシリアライズ形式（to_spec/from_spec）

- ステータス: Accepted
- 日付: 2025-09-04

## 背景
パイプラインを外部保存/再構築する標準的な表現が必要。

## 決定
JSON 互換の配列: `[{"name": str, "params": dict}]`。
- `to_spec(pipeline)` はこの形に変換。
- `from_spec(spec)` はこの形から `Pipeline` を生成（`validate_spec` を内部で呼ぶ）。

## 影響
外部ツール/設定と連携しやすく、最小限で拡張可能。

## 代替案
YAML 固有表現やクラス直列化は可搬性/安全性が低く不採用。

