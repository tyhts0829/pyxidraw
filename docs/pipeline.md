# パイプライン設計とAPI

## 目的
- 形状 `Geometry` に対する純関数的な変換の合成を、宣言的かつ安全に表現する。
- ランタイムとツール（スタブ/型検査/テスト）が同じ API 表面を共有する。

## 基本API

- ビルダー開始: `E.pipeline`
- ステップ追加: `E.pipeline.<effect>(..., **params)`
- ビルド: `.build() -> Pipeline`
- 実行: `Pipeline(g: Geometry) -> Geometry`
- 厳格モード: `.strict(enabled: bool)`（既定 True）
- キャッシュ: `.cache(maxsize: int | None)`（既定 無制限 / 0 で無効）

```
from api import G, E

# 形状 → エフェクト合成 → 実行
base = G.polygon(n_sides=6)
pipe = (
    E.pipeline
     .rotate(pivot=(200, 200, 0), angles_rad=(0, 0, 0.5))
     .fill(mode="lines", angle_rad=0.0, density=0.6)
     .build()
)
out = pipe(base)
```

## 厳格モードとパラメータ検証
- 未知キーは例外にする（strict=True）。
- `effects` は `__param_meta__` を任意で公開可能（型/範囲/choices）。
- 外部仕様（JSON など）に対する検証 API は提供しない（縮減方針）。

## 外部保存/復元（シリアライズ）
- パイプラインを JSON/ファイルに保存・復元する API は提供しない（縮減方針）。

## パフォーマンスとキャッシュ
- キーは「入力 `Geometry.digest` × パイプライン定義ハッシュ」。
- 不変パラメータのパイプラインは再利用で高速化。`.cache(maxsize=N)` で上限制御。
- `PXD_PIPELINE_CACHE_MAXSIZE` で既定値を上書き可能。

## スタブと IDE 補助
- `api/__init__.pyi` は `tools/gen_g_stubs.py` により自動生成。
- 形状 `G` とエフェクト `E.pipeline` のメソッドには引数の短い説明を含む。
- スタブは CI で同期検証（`tests/test_g_stub_sync.py`）。
