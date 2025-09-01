# PyxiDraw3

PyxiDraw3は、MIDIコントローラーを使用してリアルタイムで3Dジオメトリを生成・操作できるクリエイティブコーディングフレームワークです。

## 特徴

- **リアルタイム3Dジオメトリ生成**: 様々な3D形状（球体、多面体、トーラスなど）をリアルタイムで生成
- **MIDIコントロール**: 複数のMIDIデバイス（ArcController、TX-6、Grid、ROLIなど）に対応
- **エフェクトパイプライン**: ノイズ、回転、スケーリング、細分化などの豊富なエフェクト
- **カスタムエフェクト**: 独自のエフェクトを簡単に登録・使用可能
- **ベンチマーク機能**: パフォーマンス測定と可視化機能を内蔵
- **柔軟なAPI**: 直感的なメソッドチェーンによる操作

## インストール

```bash
git clone https://github.com/tyhts0829/pyxidraw3.git
cd pyxidraw3
pip install -r requirements.txt
```

## 基本的な使用方法

### シンプルな例

```python
import arc
from api import E, G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES

def draw(t, cc):
    # 球体を生成してエフェクトを適用
    sphere = G.sphere(subdivisions=0.5).size(80, 80, 80).at(100, 100, 0)
    sphere = E.add(sphere).noise(intensity=0.3).result()
    return sphere

if __name__ == "__main__":
    arc.start()
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"])
    arc.stop()
```

### 複雑な例（main.py）

```python
# カスタムエフェクトの登録（名前省略で関数名が登録名になる）
@E.register()
def swirl(g, strength=1.0):
    """スワール効果 - 各頂点を個別に回転"""
    # 実装詳細は main.py を参照

# 複数の形状とエフェクトを組み合わせ
def draw(t, cc):
    sphere = G.sphere(subdivisions=cc[1]).size(80, 80, 80).at(50, 50, 0)
    polygon = G.polygon(n_sides=int(cc[3] * 8 + 3)).size(60, 60, 60).at(150, 50, 0)
    
    # エフェクトチェーンの適用
    sphere_with_effects = E.add(sphere).noise(intensity=cc[5] * 0.5).filling(density=cc[6] * 0.8).result()
    polygon_with_swirl = E.add(polygon).swirl(strength=cc[7] * 2.0).wave(amplitude=cc[8] * 20.0).result()
    
    return sphere_with_effects + polygon_with_swirl

### 押し出し（Extrude）

`Extrude` は標準のエフェクトチェーンに統合されています。座標配列（coords）とオフセット配列（offsets）に対して動作し、元のライン、押し出し側のライン、両者を結ぶ側面エッジを生成します。

```python
shape = G.polygon(n_sides=6).size(80, 80, 80)
with_extrusion = E.add(shape)\
    .extrude(distance=0.2, direction=(0, 0, 1), scale=0.4)\
    .result()
```
パラメータ:
- direction: 押し出し方向ベクトル（x, y, z）
- distance: 押し出し距離係数（0.0–1.0）
- scale: 押し出し側スケール係数（0.0–1.0）
- subdivisions: 細分化係数（0.0–1.0）
```

## 主要なコンポーネント

### 形状生成 (shapes/)
- `sphere.py` - 球体生成（複数の細分化アルゴリズム対応）
- `polyhedron.py` - 正多面体（四面体、立方体、八面体など）
- `torus.py` - トーラス
- `polygon.py` - 多角形
- `text.py` - テキスト形状

### エフェクト (effects/)
- `noise.py` - ノイズエフェクト
- `rotation.py` - 回転変換
- `scaling.py` - スケーリング
- `subdivision.py` - 細分化
- `extrude.py` - 押し出し
- `filling.py` - 充填

### エンジン (engine/)
- `core/` - 基本的な幾何学処理とレンダリング
- `io/` - MIDIコントローラーとの通信
- `render/` - 3Dレンダリング

### ベンチマーク (benchmarks/)
- パフォーマンス測定
- レポート生成
- 可視化チャート

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

## テスト

```bash
# 全テストを実行
python -m pytest

# ベンチマークを実行
python -m benchmarks
```

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 著者

tyhts0829

## 貢献

プルリクエストやイシュー報告を歓迎します。
