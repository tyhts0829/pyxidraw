# ADR 0010: エフェクト関数のパラメータメタデータによる検証

- ステータス: Accepted
- 日付: 2025-09-04

## 背景
`Pipeline.validate_spec()` は JSON様式/登録名/関数シグネチャ（未知キー）を検査しているが、数値域や選択肢といった「意味的妥当性」は担保していない。実運用では UI 側からの誤入力や spec 改変を早期に検知したい。

## 決定
各エフェクト関数はオプションで `__param_meta__: dict[str, dict]` を公開できる。`validate_spec()` は存在時に最小限の検証を行う。

メタデータ項目（任意）:
- `type`: "number" | "integer" | "string"
- `min` / `max`: 数値範囲
- `choices`: 許容する離散値（列挙）

例（fill）:
```python
fill.__param_meta__ = {
  "mode": {"type": "string", "choices": ["lines", "cross", "dots"]},
  "density": {"type": "number", "min": 0.0, "max": 1.0},
}
```

## 影響
- 既存エフェクトはメタ未定義でも従来通り動作（後方互換）。
- メタを持つエフェクトは spec 事前検証が強化され、UI/保存データの堅牢性が向上。

## 代替案
- 中央定義のスキーマ管理（pydantic/attrs）: 表現力は高いが導入コストが高い。まずは軽量な辞書メタから開始する。

## 実装メモ
- `api/pipeline.validate_spec()` が `fn.__param_meta__` を見つけた場合のみ検証。
- メタはゆるい検証（実行時型）で十分。厳格な型検証は将来の拡張に委ねる。

