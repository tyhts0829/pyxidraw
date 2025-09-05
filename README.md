# PyxiDraw4

PyxiDraw4 は、MIDIコントローラーを使用してリアルタイムで 3D ジオメトリを生成・操作できるクリエイティブコーディングフレームワークです。

## 特徴

- **リアルタイム3Dジオメトリ生成**: 様々な3D形状（球体、多面体、トーラスなど）をリアルタイムで生成
- **MIDIコントロール**: 複数のMIDIデバイス（ArcController、TX-6、Grid、ROLIなど）に対応
- **エフェクトパイプライン**: ノイズ、回転、スケーリング、細分化などの豊富なエフェクト
- **カスタムエフェクト**: 独自のエフェクトを簡単に登録・使用可能
- **ベンチマーク機能**: パフォーマンス測定と可視化機能を内蔵
- **柔軟なAPI**: `E.pipeline` による関数エフェクトの組み立て

## インストール

依存は用途別に分かれます。まずは最小構成を入れ、必要に応じて機能別オプションを追加してください。

必須（最小構成）:

```bash
git clone <this-repo>
cd pyxidraw4
pip install numpy numba pyglet moderngl psutil pyyaml
```

オプション（機能別）:

- バッファ/オフセット系（Shapely を使用）
  - `pip install shapely`
- アセミックグリフ（KD-Tree で SciPy を使用）
  - `pip install scipy`
- MIDI コントローラ（mido）
  - `pip install mido python-rtmidi`（環境依存）
  - MIDI は既定で有効です。無効化したい場合は `run(..., use_midi=False)` または `python main.py --no-midi` を使用してください。
  - デバイス未接続／依存未導入時は警告のうえ自動フォールバック（NullMidi）します。厳格に失敗させたい場合は `--midi-strict` か `PYXIDRAW_MIDI_STRICT=1` を利用してください。

## 基本的な使用方法

### シンプルな例（関数エフェクト + パイプライン）

```python
from api import E, G, run
from util.constants import CANVAS_SIZES

def draw(t, cc):
    # 球体を生成してエフェクトを適用（MIDI不要）
    sphere = G.sphere(subdivisions=0.5).scale(80, 80, 80).translate(100, 100, 0)
    pipeline = E.pipeline.displace(amplitude_mm=0.3).build()
    return pipeline(sphere)

if __name__ == "__main__":
    run(draw, canvas_size=CANVAS_SIZES["SQUARE_200"]) 
```

MIDI は既定で有効です。無効化する場合は `run(..., use_midi=False)` もしくは `python main.py --no-midi` を使用してください。

### MIDI フォールバックと厳格モード

- 既定: 有効化時にデバイスや依存が見つからない場合、警告のみで NullMidi にフォールバック（例外は出ません）。
- 厳格モード: `python main.py --midi-strict` または `PYXIDRAW_MIDI_STRICT=1` で有効。初期化失敗時は `SystemExit(2)` を返します。
- 環境変数の真偽解釈: `1/true/on/yes` → True、`0/false/off/no` → False。
- 設定ファイルで既定を変更: `config.yaml` の `midi.enabled_default` / `midi.strict_default`。

#### ヘッドレス実行（Headless）
- 目的: ウィンドウや OpenGL を起動せずに、描画関数だけを実行して Geometry の統計を確認。
- 有効化: 環境変数 `PYXIDRAW_HEADLESS=1` を付けて実行します。

```bash
PYXIDRAW_HEADLESS=1 python tutorials/01_basic_shapes.py
```

- 挙動: `draw(0, {})` を1回だけ呼び、頂点数・ライン数をログ出力（例: `Headless OK: points=..., lines=...`）。
- ユースケース: CI/サーバーでのスモークチェック、依存の軽い実行確認。
- 注意: レンダリング品質や見た目は検証しません。時間依存の確認は `t` を変えて繰り返し `draw(t, {})` を呼び出してください。

### 角度・スケールの取り扱い（指針）

- 角度入力は `angles_rad`（ラジアン）を明示してください（`effects.rotation/transform`）。0..1 の暗黙指定や `angles_deg` は使用しません。
- `translation` は物理単位（mm）を直接指定。
- `scaling` は `(sx, sy, sz)` の倍率指定。スカラー/1要素/3要素を受け付けます。

命名の推奨:
- 中心は `pivot` を使用します。
- 角度は `angles_rad` を明示します（0..1 の暗黙指定は非推奨）。

### 複雑な例（main.py）

```python
# 関数エフェクトと新パイプライン（詳細は main.py を参照）

# 複数の形状とエフェクトを組み合わせ
def draw(t, cc):
    sphere = G.sphere(subdivisions=cc[1]).scale(80, 80, 80).translate(50, 50, 0)
    polygon = G.polygon(n_sides=int(cc[3] * 8 + 3)).scale(60, 60, 60).translate(150, 50, 0)
    
    # エフェクトパイプラインの適用
    sphere_with_effects = (E.pipeline
                            .displace(amplitude_mm=cc[5] * 0.5)
                            .fill(density=cc[6] * 0.8)
                            .build())(sphere)
    polygon_with_swirl = polygon  # 例の簡略化
    
    return sphere_with_effects + polygon_with_swirl

### 押し出し（Extrude）

`effects.extrude` は関数エフェクトとして実装済みです。座標配列（coords）とオフセット配列（offsets）に対して動作し、元のライン、押し出し側のライン、両者を結ぶ側面エッジを生成します。

```python
from api import E, G
base = G.polygon(n_sides=6).scale(80, 80, 80)
out = (E.pipeline
         .extrude(direction=(0, 0, 1), distance=0.5, scale=1.0, subdivisions=0.3, center_mode="auto")
         .build())(base)
```
主なパラメータ:
- direction: 押し出し方向ベクトル（x, y, z）
- distance: 押し出し距離係数（0.0–1.0）
- scale: 押し出し側スケール係数（0.0–1.0）
- subdivisions: 細分化係数（0.0–1.0）
- center_mode: 押し出し側スケールの中心（"origin"|"auto"）。"auto" は押し出し先ラインの重心を基準にスケール。

## 主要なコンポーネント

### 形状生成 (shapes/)
- `sphere.py` - 球体生成（複数の細分化アルゴリズム対応）
- `polyhedron.py` - 正多面体（四面体、立方体、八面体など）
- `torus.py` - トーラス
- `polygon.py` - 多角形
- `text.py` - テキスト形状

### エフェクト (effects/)
- `noise.py` - ノイズ
- `rotation.py` - 回転
- `translation.py` - 平行移動
- `scaling.py` - スケーリング
- `transform.py` - 複合変換
- `filling.py` - 充填（ハッチ/ドット）
- `array.py` `dashify.py` `extrude.py` `twist.py` `explode.py` `wobble.py` など

パラメータ指針（抜粋）:
- 正規化系: 0..1 を受け取り内部でレンジに写像
  - 例: `rotate.angles_rad`（ラジアン）, `extrude.distance/subdivisions`（0..1→上限レンジ）, `offset.distance/segments_per_circle`
- 物理/実値系: 座標単位（mm相当）・そのままの値
  - 例: `translate.delta`, `dash.dash_length/gap_length`, `ripple.amplitude`, `wobble.amplitude`
- 空間周波数: `wave/wobble.frequency` は「座標1あたりの周期数 [cycles per unit]」

### エンジン (engine/)
- `core/` - 基本的な幾何学処理とレンダリング
- `io/` - MIDIコントローラーとの通信
- `render/` - 3Dレンダリング

### ベンチマーク (benchmarks/)
- パフォーマンス測定（`python -m benchmarks run`）
- ターゲット一覧/絞り込み（`list --tag`, `--plugin`）
- 結果比較（`compare --abs-threshold`、タグ/ターゲット別しきい値）
- 失敗のみ再実行（`run --from-file benchmark_results/failed_targets.json`）
- 個別スキップ（`run --skip effects.noise.high_frequency`）
- レポート生成（HTML/Markdown、自動出力）

## 設定

`config.yaml`でMIDIデバイスやレンダリング設定をカスタマイズできます。

```yaml
canvas:
  background_color: [1.0, 1.0, 1.0, 1.0]
canvas_controller:
  fps: 24
midi:
  enabled_default: true
  strict_default: false
midi_devices:
  - port_name: "ArcController OUT"
    mode: "14bit"
    controller_name: "arc"
```

補足: `midi.enabled_default` と `midi.strict_default` で既定の有効化/厳格モードの挙動を切り替えできます。

## パイプラインのシリアライズ/検証

`to_spec` と `from_spec` でパイプラインを保存/復元できます。`validate_spec` で事前検証も可能です。

```python
from api import E, to_spec, from_spec, validate_spec

pipeline = (E.pipeline.rotate(angles_rad=(0.5 * 3.141592653589793, 0.0, 0.0))
                       .displace(amplitude_mm=0.2)
                       .build())
spec = to_spec(pipeline)
validate_spec(spec)     # 構造/登録名/パラメータ検証（例外が出なければOK）
pipeline2 = from_spec(spec)
```

効果のパラメータ仕様は `docs/effects_cheatsheet.md` を参照してください。
アーキテクチャ決定（ADR）は `docs/adr/README.md` を参照してください。

## テスト

```bash
# 全テストを実行
python -m pytest

# ベンチマークを実行（全ターゲット）
python -m benchmarks run
```

## 開発向け: エフェクトの追加/作成

- エフェクトは純関数（`Geometry -> Geometry`）として `effects/` に追加し、`@effects.registry.effect()` で登録します。
- 仕様検証のため、任意で `__param_meta__`（型/範囲/choices）を関数属性として宣言できます。
- 詳細は「docs/guides/effects_authoring.md」を参照してください。

## キャッシュ制御（開発向け）

形状生成の LRU キャッシュは環境変数で制御できます。

```bash
# キャッシュ無効化
export PXD_CACHE_DISABLED=1

# キャッシュサイズ上書き（デフォルト128）
export PXD_CACHE_MAXSIZE=64
```

パイプライン（`api.pipeline.Pipeline`）の単層キャッシュも環境変数で制御できます。

```bash
# パイプラインキャッシュの最大保持数（LRU 風）。0 で無効、未設定(None)で無制限。
export PXD_PIPELINE_CACHE_MAXSIZE=128
```

ジオメトリダイジェスト（`Geometry.digest`）は環境変数で無効化できます。無効化時でもパイプラインのキャッシュはフォールバックハッシュで機能します。

```bash
# Geometry のダイジェスト計算を無効化（ベンチ比較や計測向け）
export PXD_DISABLE_GEOMETRY_DIGEST=1
```

補足:
- `PXD_CACHE_*`: 形状生成（`shapes/*`）の LRU キャッシュ制御。
- `PXD_PIPELINE_CACHE_MAXSIZE`: パイプラインの単層キャッシュ上限（0 で無効）。
- `PXD_DISABLE_GEOMETRY_DIGEST`: Geometry のダイジェスト保持を無効化（パイプラインはフォールバックで継続）。

### 登録済みエフェクトの確認（開発向け）

現在レジストリに登録されているエフェクト名を一覧できます。

```python
>>> from effects.registry import list_effects
>>> list_effects()[:8]
['repeat', 'collapse', 'dash', 'explode', 'extrude', 'fill', 'offset', 'rotate']
```

スクリプト内からの防御的プログラミングにも利用できます（未登録名の早期検出など）。

## ベンチマーク ワークフロー（基準化）

結果を保存して比較する最小フロー例:

```bash
# 基準（baseline）を保存
python -m benchmarks run -o benchmark_results/baseline

# 変更後の結果を保存
python -m benchmarks run -o benchmark_results/current

# 差分を比較
python -m benchmarks compare benchmark_results/baseline/latest.json benchmark_results/current/latest.json

# しきい値を調整（2ms未満は無視、alloc-heavy の回帰閾値を厳しめに）
python -m benchmarks compare \
  benchmark_results/baseline/latest.json benchmark_results/current/latest.json \
  --abs-threshold 0.002 --tag alloc-heavy
```

高速なスモーク実行（保存/チャート無効）:

```bash
python -m benchmarks run --warmup 0 --runs 1 --timeout 15 --no-charts --no-save

タグで絞り込んだ実行や一覧:

```bash
# タグで一覧
python -m benchmarks list --tag cpu-bound --format table

# 失敗のみ再実行
python -m benchmarks run --from-file benchmark_results/failed_targets.json

# 個別ターゲットをスキップ
python -m benchmarks run --skip effects.noise.high_frequency --skip shapes.sphere.high_res
```

並列実行の目安は docs/benchmarks_parallel_guide.md を参照してください。
```

## データ形式（npz 統一）

このリポジトリに含まれる幾何データ資産はすべて `.npz` に統一されています。
`shapes/polyhedron.py` を含むランタイムは `.npz` のみを対象とし、`.pkl` は使用しません。

既存の外部 pickle 資産を `.npz` に変換したい場合は、同梱の変換スクリプトを利用できます（任意・外部資産向け）。

```bash
# 外部 pickle → npz 変換（dry-run / 実行 / 削除オプション）
python scripts/convert_polyhedron_pickle_to_npz.py --dry-run --verbose
python scripts/convert_polyhedron_pickle_to_npz.py --verbose
python scripts/convert_polyhedron_pickle_to_npz.py --delete-original --verbose
```

理由（要約）:
- 安全性（pickle は任意コード実行リスク、npz は純データ）
- 互換性（環境非依存）/再現性（dtype/shape 決定的）
- 実装の単純化

現状確認:
```bash
ls data/regular_polyhedron/*_vertices_list.npz
rg "_vertices_list\\.pkl" -n data || echo "OK: no .pkl files in repo"
```

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 著者

tyhts0829

## 貢献

プルリクエストやイシュー報告を歓迎します。
