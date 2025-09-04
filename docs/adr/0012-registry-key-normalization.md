# ADR 0012: Registry Key Normalization Policy

Date: 2025-09-04

## Status
Accepted

## Context
`effects.registry` と `shapes.registry` で名前正規化の微差があった（ハイフン置換/小文字化 vs CamelCase→snake 変換を含むルール）。
呼び出しの一貫性と学習コスト削減のため、単一ポリシーに統一する。

## Decision
- キー正規化は `common.base_registry.BaseRegistry._normalize_key()` に集約。
- 具体的には以下を適用する:
  - `-` → `_` に置換
  - CamelCase → snake_case 変換（必要時）
  - 小文字化
- `effects.registry` も上記に従う（実装済: BaseRegistry のメソッドを呼ぶ）。

## Consequences
- `RotateZ`/`rotateZ`/`rotate-z`/`rotate_z` が同一キーとして扱われ、利用者体験が一貫化。
- 旧挙動からの破壊的変更はなく、より受容範囲が広がる方向の変更。

## Related
- docs/simplicity_readability_audit_2025-09-04.md（レジストリ/命名の一貫性）
