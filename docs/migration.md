# 移行ガイド（v3 → v4 クリーンAPI）

本ガイドは、後方互換や非推奨エイリアスを廃止した v4 系APIへの移行のための最小置換表です。説明や猶予期間は設けず、置換に失敗した箇所は実行時に例外として検出されます（`PipelineBuilder` 既定 strict=True, `validate_spec` も未知キーをエラー）。

## 置換表（必須）

- テキスト形状
  - `G.text(..., size=...)` → `G.text(..., font_size=...)`

- ノイズ系エフェクト（displace）
  - `displace(intensity=...)` → `displace(amplitude_mm=...)`
  - `displace(frequency=...)` → `displace(spatial_freq=...)`
  - `displace(time=...)` → `displace(t_sec=...)`

## パイプラインの厳格化

- `E.pipeline` は既定で厳格検証（unknown params は TypeError）。明示する場合は `.strict(True)` を付与できます。
- 仕様検証 `validate_spec(spec)` も、登録名とパラメータ名を厳密に突き合わせ、未知キーを `TypeError` とします。

## キャッシュ設定の補足

- パイプライン単層キャッシュの上限は環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` で制御できます（未設定=無制限、`0`=無効）。

## サンプル

```python
from api import E, G, validate_spec, to_spec, from_spec

g = G.sphere(subdivisions=0.5).scale(100,100,100).translate(100,100,0)
pipeline = (E.pipeline
              .displace(amplitude_mm=0.2, spatial_freq=0.5, t_sec=0.0)
              .strict(True)
              .build())

spec = to_spec(pipeline)
validate_spec(spec)
result = from_spec(spec)(g)
```

