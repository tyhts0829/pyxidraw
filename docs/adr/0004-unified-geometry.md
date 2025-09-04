# ADR 0004: Geometry 型の一本化

- ステータス: Accepted
- 日付: 2025-09-04

## 背景
`GeometryData` など複数のデータ表現が併存すると、エフェクト/描画/シリアライズの境界で摩擦が生じる。

## 決定
`engine/core/geometry.py` の `Geometry(coords, offsets)` に一本化。変換は `translate/scale/rotate/concat` の純関数（新インスタンス返却）。

## 影響
API の心的モデルが単純に。バッファ生成・描画も一本化される。

## 代替案
軽量ラッパ/ビュー案もあったが、まず一本化で認知負荷を下げる。

