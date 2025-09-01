# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# PyxiDraw3 プロジェクト仕様書

## プロジェクト概要

PyxiDraw3 は、MIDI コントローラーを使用してリアルタイムで 3D ジオメトリを生成・操作できるクリエイティブコーディングフレームワークです。アーティストやデザイナーが直感的に 3D 形状を操作し、ビジュアル表現を行うためのツールです。

### 主な特徴

- リアルタイム 3D ジオメトリ生成
- 複数の MIDI デバイス対応（ArcController、TX-6、Grid、ROLI 等）
- 豊富なエフェクトパイプライン
- パフォーマンスベンチマークシステム
- カスタムエフェクトの簡単な登録

## 技術スタック

### 言語・フレームワーク

- Python 3.x
- NumPy（数値計算）, Numba
- arc（MIDI インターフェース）
- pyglet/moderngl（3D レンダリング - 推定）
- matplotlib（ベンチマーク可視化）
- YAML（設定ファイル）

### 外部依存関係

注意: requirements.txt が見つからないため、以下はコードから推定される主要な依存関係です：

```bash
# 必須パッケージのインストール
pip install numpy pyyaml matplotlib seaborn
# arc は外部ライブラリのため別途インストールが必要
```

## プロジェクト構造

```
pyxidraw3/
├── api/                    # 高レベルAPI
│   ├── effect_chain.py     # エフェクトチェーン
│   ├── effect_pipeline.py  # エフェクトパイプライン
│   ├── geometry_api.py     # ジオメトリAPI
│   ├── runner.py          # スケッチ実行エンジン
│   ├── shape_factory.py   # 形状生成ファクトリー
│   └── shape_registry.py  # 形状レジストリ
├── shapes/                # 3D形状定義
│   ├── sphere.py         # 球体（複数の細分化アルゴリズム）
│   ├── polyhedron.py     # 正多面体
│   ├── torus.py          # トーラス
│   ├── polygon.py        # 多角形
│   ├── cylinder.py       # シリンダー
│   ├── cone.py           # 円錐
│   ├── grid.py           # グリッド
│   ├── text.py           # テキスト形状
│   └── ...
├── effects/              # エフェクト実装
│   ├── noise.py         # ノイズエフェクト
│   ├── rotation.py      # 回転
│   ├── scaling.py       # スケーリング
│   ├── subdivision.py   # 細分化
│   ├── extrude.py       # 押し出し
│   ├── filling.py       # 充填
│   ├── array.py         # 配列複製
│   └── ...
├── engine/              # コアエンジン
│   ├── core/           # コア機能
│   ├── io/             # MIDI I/O
│   ├── render/         # レンダリング
│   ├── pipeline/       # 処理パイプライン
│   └── ui/             # UI要素
├── benchmarks/         # ベンチマークシステム
│   ├── core/          # ベンチマークコア
│   ├── plugins/       # プラグインシステム
│   ├── visualization/ # 結果可視化
│   └── tests/         # ベンチマークテスト
├── data/              # プリセットデータ
│   ├── sphere/        # 球体頂点データ
│   └── regular_polyhedron/ # 正多面体データ
├── tests/             # ユニットテスト
├── main.py           # メインデモ（複雑な例）
├── simple.py         # シンプルデモ
└── config.yaml       # グローバル設定

benchmark_results/    # ベンチマーク結果の出力先
```

## 主要コンポーネント

### 1. API 層（api/）

- **GeometryAPI**: ジオメトリ操作の基本クラス
- **ShapeFactory (G)**: 形状生成のファクトリーインターフェース
- **EffectChain (E)**: エフェクトチェーンの構築と実行
- **runner**: スケッチ実行のメインループ管理

### 2. 形状システム（shapes/）

- 基本形状の定義と生成ロジック
- `BaseShape`を継承した各種形状クラス
- レジストリパターンによる動的登録

### 3. エフェクトシステム（effects/）

- `BaseEffect`を継承したエフェクト実装
- パイプラインパターンによる連鎖処理
- カスタムエフェクトの登録機能（`@E.register()` デコレータ。関数名から自動登録）

補足:
- 標準エフェクトは `@effect` デコレータ（省略名推奨）で登録され、`EffectChain` は中央レジストリから解決します。
- `Extrude` を含む全エフェクトは coords/offsets ベースで動作します（`GeometryAPI` レイヤにより統一）。

### 4. エンジン（engine/）

- **core/**: ジオメトリデータ構造、変換ユーティリティ
- **io/**: MIDI 入力の処理とマッピング
- **render/**: 3D レンダリングシステム
- **pipeline/**: 非同期処理パイプライン

### 5. ベンチマークシステム（benchmarks/）

- プラグインベースの拡張可能なベンチマーク
- 自動レポート生成（HTML/Markdown）
- パフォーマンス可視化（グラフ、ヒートマップ）

## コマンドリファレンス

### 実行コマンド

```bash
# メインデモの実行（複雑な例）
python main.py

# シンプルなデモの実行  
python simple.py

# MIDIなしで実行（テスト用）
python main.py --no-midi
```

### テストコマンド

```bash
# 全テスト実行
python -m pytest

# 特定のテストファイル実行
python -m pytest tests/test_shapes.py

# 特定のテスト関数実行
python -m pytest tests/test_shapes.py::test_sphere

# カバレッジ付きテスト
python -m pytest --cov=. --cov-report=html
```

### ベンチマークコマンド

```bash
# 利用可能なベンチマーク一覧
python -m benchmarks list

# 全ベンチマーク実行
python -m benchmarks run

# 特定のベンチマーク実行
python -m benchmarks run --target shapes.sphere

# 結果の比較
python -m benchmarks compare results/old.json results/new.json

# HTMLレポート生成
python -m benchmarks report --format html
```

## 使用方法

### 基本的な使い方

```python
import arc
from api import E, G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES

def draw(t, cc):
    # 形状の生成
    sphere = G.sphere(subdivisions=0.5).size(80, 80, 80).at(100, 100, 0)

    # エフェクトの適用
    sphere = E.add(sphere).noise(intensity=0.3).result()

    return sphere

if __name__ == "__main__":
    arc.start()
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"])
    arc.stop()
```

### カスタムエフェクトの登録

```python
@E.register()
def custom_effect(g, parameter=1.0):
    """カスタムエフェクトの実装"""
    # ジオメトリを操作
    coords = g.coords.copy()
    # ... 処理 ...
    return GeometryAPI.from_lines(processed_lines)
```

## MIDI 設定

`config.yaml`で MIDI デバイスのマッピングを設定：

```yaml
midi_devices:
  - port_name: "ArcController OUT"
    mode: "14bit"
    controller_name: "arc"
    cc_map:
      1: 1
      2: 2
      # ...
```

## ベンチマーク実行

```bash
# 全ベンチマーク実行
python -m benchmarks run

# 特定のターゲットのみ
python -m benchmarks run --target effects.noise

# 結果の比較
python -m benchmarks compare old.json new.json
```

## 開発上の注意点

### コーディング規約

1. NumPy 配列は`dtype=np.float32`を使用（パフォーマンス考慮）
2. ジオメトリデータは`coords`（頂点座標）と`offsets`（線分境界）で管理
3. エフェクトは非破壊的に実装（元のジオメトリを変更しない）
4. レジストリパターンを使用した動的登録

### パフォーマンス考慮

1. 大規模なジオメトリ処理では NumPy のベクトル化演算を活用
2. キャッシュ機能（`cacheable_base.py`）を適切に使用
3. ベンチマークシステムでパフォーマンスを定期的に確認

### エラーハンドリング

1. MIDI 接続エラーは適切にハンドリング（`midi=False`でテスト可能）
2. ジオメトリ操作の境界条件をチェック
3. エフェクトパラメータの妥当性検証

## アーキテクチャ詳細

### ジオメトリデータ構造

```python
# GeometryAPIの内部構造
class GeometryAPI:
    coords: np.ndarray  # 頂点座標 (N, 3) float32
    offsets: np.ndarray  # 線分境界インデックス
    colors: np.ndarray   # 頂点カラー (optional)
    normals: np.ndarray  # 法線ベクトル (optional)
```

### エフェクトチェーンの仕組み

```python
# メソッドチェーンによる連続適用
result = E.add(geometry)
    .noise(intensity=0.5)     # ノイズ追加
    .rotate(x=45, y=30)       # 回転
    .subdivide(iterations=2)  # 細分化
    .result()                 # 最終結果取得
```

### MIDIマッピング構造

```python
# CCマッピング例
cc_values = {
    1: 0.5,   # ノイズ強度
    2: 0.3,   # 回転角度
    3: 0.7,   # スケール
    # ...
}
```

## トラブルシューティング

### よくある問題

1. **MIDI 接続エラー**

   - `arc.start(midi=False)`で MIDI を無効化してテスト
   - `config.yaml`でデバイス名を確認

2. **レンダリングパフォーマンス**

   - `subdivisions`パラメータを調整
   - `render_scale`を下げて確認

3. **メモリ使用量**
   - 大規模なジオメトリでは細分化レベルに注意
   - ベンチマーク結果でメモリ使用量を監視

## 今後の開発方向

1. **拡張予定の機能**

   - より多くの形状タイプ
   - 高度なエフェクト（物理シミュレーション等）
   - リアルタイムプレビューの改善

2. **最適化の余地**
   - GPU 活用の拡大
   - 並列処理の強化
   - メモリ効率の改善

## リソース

- プロジェクトリポジトリ: [GitHub URL]
- 作者: tyhts0829
- ライセンス: MIT

## 開発ワークフロー

### 新しい形状の追加

1. `shapes/`に新しいファイルを作成
2. `BaseShape`を継承
3. `generate()`メソッドを実装
4. `shape_registry.py`に登録
5. テストを`tests/test_shapes.py`に追加

### 新しいエフェクトの追加

1. `effects/`に新しいファイルを作成
2. `BaseEffect`を継承または`@E.register`デコレータ使用
3. エフェクト関数を実装
4. テストを`tests/test_effects.py`に追加
5. ベンチマークを`benchmarks/plugins/`に追加（推奨）

## 備考

- `requirements.txt`が見つからないため、依存関係は手動でインストールが必要
- `.gitignore`や`.github/workflows`などの CI/CD 設定は現在未実装
- プロジェクトは活発に開発中で、API は変更される可能性あり
- lintやformatのコマンドは現在未定義（flake8、black等の設定ファイルなし）
