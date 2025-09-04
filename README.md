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

依存関係は固定の requirements.txt を設けていません（提案に基づく最小依存）。以下は最小構成の例です。

```bash
git clone <this-repo>
cd pyxidraw4
pip install numpy numba pyglet moderngl psutil pyyaml
```

## 基本的な使用方法

### シンプルな例（関数エフェクト + パイプライン）

```python
import arc
from api import E, G, run
from util.constants import CANVAS_SIZES

def draw(t, cc):
    # 球体を生成してエフェクトを適用
    sphere = G.sphere(subdivisions=0.5).scale(80, 80, 80).translate(100, 100, 0)
    pipeline = E.pipeline.noise(intensity=0.3).build()
    return pipeline(sphere)

if __name__ == "__main__":
    arc.start()
    run(draw, canvas_size=CANVAS_SIZES["SQUARE_200"]) 
    arc.stop()
```

### 角度・スケールの取り扱い（指針）

- 角度入力は 0..1 を基本とし、内部で 0..2π に正規化（`effects.rotation/transform`）。
- `translation` は物理単位（mm）を直接指定。
- `scaling` は `(sx, sy, sz)` の倍率指定。スカラー/1要素/3要素を受け付けます。

### 複雑な例（main.py）

```python
# 関数エフェクトと新パイプライン（詳細は main.py を参照）

# 複数の形状とエフェクトを組み合わせ
def draw(t, cc):
    sphere = G.sphere(subdivisions=cc[1]).scale(80, 80, 80).translate(50, 50, 0)
    polygon = G.polygon(n_sides=int(cc[3] * 8 + 3)).scale(60, 60, 60).translate(150, 50, 0)
    
    # エフェクトパイプラインの適用
    sphere_with_effects = (E.pipeline
                            .noise(intensity=cc[5] * 0.5)
                            .filling(density=cc[6] * 0.8)
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
  - 例: `rotation.rotate`（0..1→2π）, `extrude.distance/subdivisions`（0..1→上限レンジ）, `buffer.distance/resolution`
- 物理/実値系: 座標単位（mm相当）・そのままの値
  - 例: `translation.offset_*`, `dashify.dash_length/gap_length`, `wave.amplitude`, `wobble.amplitude`
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
midi_devices:
  - port_name: "ArcController OUT"
    mode: "14bit"
    controller_name: "arc"
```

## パイプラインのシリアライズ/検証

`to_spec` と `from_spec` でパイプラインを保存/復元できます。`validate_spec` で事前検証も可能です。

```python
from api import E, to_spec, from_spec, validate_spec

pipeline = (E.pipeline.rotation(rotate=(0.25,0,0)).noise(intensity=0.2).build())
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

## キャッシュ制御（開発向け）

形状生成の LRU キャッシュは環境変数で制御できます。

```bash
# キャッシュ無効化
export PXD_CACHE_DISABLED=1

# キャッシュサイズ上書き（デフォルト128）
export PXD_CACHE_MAXSIZE=64
```

### 登録済みエフェクトの確認（開発向け）

現在レジストリに登録されているエフェクト名を一覧できます。

```python
>>> from effects.registry import list_effects
>>> list_effects()[:8]
['array', 'boldify', 'buffer', 'collapse', 'dashify', 'explode', 'extrude', 'filling']
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

## データ移行（polyhedron: pickle → npz）

正多面体データは順次 `.npz` 形式へ移行しています（`shapes/polyhedron.py` は `.npz` を優先読み込み）。
既存の `data/regular_polyhedron/*_vertices_list.pkl` を一括変換するには次のスクリプトを使用します。

```bash
# 変換（dry-run）
python scripts/convert_polyhedron_pickle_to_npz.py --dry-run --verbose

# 実変換（上書きしない）
python scripts/convert_polyhedron_pickle_to_npz.py --verbose

# 変換後に .pkl を削除
python scripts/convert_polyhedron_pickle_to_npz.py --delete-original --verbose

# 既存の .npz を上書き
python scripts/convert_polyhedron_pickle_to_npz.py --force
```

なぜ npz?:
- 安全性（pickleは任意コード実行リスク、npzは純データ）
- 互換性（環境非依存で将来の移行が容易）
- 再現性（dtype/shape/順序が決定的）
- 単純化（ローダ/実装が簡潔）

備考:
- 何度実行しても安全です（既存 `.npz` は既定では上書きしません）。
- 変換後は `.pkl` を削除して構いません（`--delete-original` で自動削除可）。
- 詳細な判断理由は `docs/proposals/completed/PROPOSAL_BREAKING_CHANGES.md` の「決定記録」および
  ADR: `docs/adr/0001-npz-over-pickle.md` を参照してください。

移行完了チェック:
```bash
# .npz が存在し .pkl が無いことを確認
ls data/regular_polyhedron/*_vertices_list.npz
rg "_vertices_list\\.pkl" -n data/regular_polyhedron || echo "OK: no .pkl files"
```

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 著者

tyhts0829

## 貢献

プルリクエストやイシュー報告を歓迎します。
