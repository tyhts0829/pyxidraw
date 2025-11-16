# パイプライン設計とAPI

## 目的
- 形状 `Geometry` に対する純関数的な変換の合成を、宣言的かつ安全に表現する。
- ランタイムとツール（スタブ/型検査/テスト）が同じ API 表面を共有する。

## 基本API

- ビルダー開始: `E.<effect>(..., **params)` または `E.pipeline`
- ステップ追加: `E.<effect>(..., **params)` をメソッドチェーンで連ねる（`E.pipeline.<effect>(..., **params)` も利用可能）
- 表示ラベル: `.label(uid: str)`（Parameter GUI のカテゴリ名を指定。内部 UID は維持）
- ビルド: `.build() -> Pipeline`
- 実行: `Pipeline(g: Geometry) -> Geometry`
- キャッシュ: `.cache(maxsize: int | None)`（既定 無制限 / 0 で無効）

```
from api import G, E

# 形状 → エフェクト合成 → 実行
base = G.polygon(n_sides=6)
pipe = (
    E.rotate(pivot=(200, 200, 0), angles_rad=(0, 0, 0.5))
     .fill(angle_sets=1, angle_rad=0.0, density=0.6)
     .build()
)
out = pipe(base)
```

注記（label）
- `.label(uid)` はパイプラインの「表示ラベル」を設定する。GUI のカテゴリ見出しに用いられる。
- 内部のパイプライン UID（例: `p0`, `p1`）は保持し、Parameter ID/キャッシュ鍵/署名には影響しない。
- 重複ラベルは許容（カテゴリ見出しは同名でまとめて表示され得る）。

## パラメータ検証
- ビルド時の厳格検証は行わない（未知キーは許容）。実行時にエフェクト関数のシグネチャで自然に TypeError となり得る。
- `effects` は `__param_meta__` を任意で公開可能（型/範囲/choices）。
- 外部仕様（JSON など）に対する検証 API は提供しない（縮減方針）。

## 外部保存/復元（シリアライズ）
- パイプラインを JSON/ファイルに保存・復元する API は提供しない（縮減方針）。

## パフォーマンスとキャッシュ
- キーは `api.lazy_signature.lazy_signature_for(LazyGeometry)` に基づく 128bit 署名。
  - base が実体 `Geometry` の場合はオブジェクト同一性（`("geom-id", id(g))`）を用い、内容ダイジェストは計算しない。
  - base が shape の場合は `impl_id(shape)` と `params_signature` を用いる。
  - plan（effect 列）は各ステップの `impl_id(effect)` と `params_signature` を順に積んで署名に反映する。
- 不変パラメータのパイプラインは再利用で高速化。`.cache(maxsize=N)` で上限制御（`None` は無制限、`0` は無効）。
- `PXD_PIPELINE_CACHE_MAXSIZE` で既定値を上書き可能。

### 量子化と鍵の安定化（重要）
- パラメータは署名生成前に量子化される（`common.param_utils.params_signature`）。
  - 対象は float（`float | np.floating`）のみ。int/bool は非量子化。
  - 粒度は `__param_meta__['step']` を優先し、未指定時の既定は 1e-6。環境変数 `PXD_PIPELINE_QUANT_STEP` で既定を上書き可能。
  - ベクトルは成分ごとに適用。`step` がタプル/配列で短い場合は末尾値で補完。
- Effects は量子化後の値がそのまま実行時の引数になる（再現性と安定性を優先）。
- Shapes はキャッシュ鍵の生成に量子化を使うが、実行にはランタイム解決後の値を渡す（UI と直呼びの一貫性を優先）。

## スタブと IDE 補助
- `api/__init__.pyi` は `tools/gen_g_stubs.py` により自動生成。
- 形状 `G` とエフェクト `E.<effect>`/`E.pipeline.<effect>` のメソッドには引数の短い説明を含む。
- スタブは CI で同期検証（`tests/test_g_stub_sync.py`）。

## 関連ユーティリティ
- 時間変調（LFO）: パイプライン外の補助として `from api import lfo` を提供。詳細は `docs/lfo_spec.md` を参照。LFO は時間 `t` を入力に 0..1 の値を返し、効果パラメータや形状引数の連続変化に用いる。
