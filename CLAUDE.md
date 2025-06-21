# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PyxiDraw - MIDI-Controlled Vector Graphics System

PyxiDraw は、MIDI コントローラーと連携したリアルタイム 2D/3D ベクトルグラフィックス生成システムです。AxiDraw などのペンプロッターデバイスへの出力にも対応しています。

## 開発コマンド

### 基本実行

```bash
python main.py                    # 基本デモの実行
python all_shapes.py             # 全形状の展示デモ
python -m pytest tests/          # テスト実行（存在する場合）
```

### MIDI 環境

MIDI デバイスが接続されている場合:

```python
from api.runner import run_sketch
from engine.io import arc

arc.start(midi=True)  # MIDI初期化
```

## アーキテクチャ概要

### 4 層アーキテクチャ

1. **API 層** (`/api/`): 高レベルユーザー API

   - `runner.py`: スケッチ実行環境の統合管理
   - `shapes.py`: 形状生成の統一 API
   - `effects.py`: エフェクト適用の統一 API

2. **エンジン層** (`/engine/`): コアシステム

   - `core/`: フレーム管理、レンダリングウィンドウ
   - `io/`: MIDI 入力処理、コントローラー管理
   - `pipeline/`: 並列処理、データストリーミング
   - `render/`: OpenGL ベース線画レンダラー

3. **形状・エフェクト層** (`/shapes/`, `/effects/`):

   - 豊富な 2D/3D 形状生成（基本図形、正多面体、リサージュ曲線、ストレンジアトラクターなど）
   - 変形エフェクト（boldify、noise、subdivision、connect 等）

4. **ユーティリティ層** (`/util/`):
   - 数学関数、定数定義、幾何学計算

### データフロー

```
MIDI入力 → IO処理 → スケッチ関数(draw) → 形状生成 → エフェクト適用 → レンダリング → 出力
```

## 重要な設計原則

### スケッチ関数パターン

```python
def draw(t: float, cc: Dict[int, float]) -> Lines:
    # t: 時間 (seconds)
    # cc: MIDI CC values
    return shape_function() | effect_function()
```

### 型安全性

- 全モジュールで型ヒント使用
- `Lines`型: ベクター線データの標準表現
- NumPy 配列ベースの効率的な数値計算
- np.float32 を基本とする
- effect の引数は 0.0-1.0 のレンジを想定し、クラス定数で適切にスケーリングする。デフォルト値はすべて 0.5 とする。

### 並列処理

- `engine/pipeline/`: ワーカープールによるバックグラウンド計算
- GPU 加速レンダリング（ModernGL）

## MIDI デバイス対応

### サポートデバイス

- TE OP-Z, TX-6
- monome Grid, Arc
- 一般的な MIDI CC コントローラー

### MIDI CC 使用例

```python
def draw(t, cc):
    rotation_speed = cc.get(74, 0.5)  # CC#74でローテーション制御
    scale = cc.get(71, 1.0)          # CC#71でスケール制御
    return polygon(6) | rotate(t * rotation_speed) | scale_by(scale)
```

## データ管理

### 事前計算データ

- `/data/`: 3D モデルの頂点・面データ（.npz ファイル）
- 最適化された正多面体データ

### 設定ファイル

- `util/constants.py`: キャンバスサイズ、ノイズパラメータ
- `previous_design/src/config.yaml`: MIDI 設定（参考用）

## 開発時の注意点

### スレッドセーフティ

- IO モジュールで同期処理に注意
- MIDI 入力は別スレッドで処理される

### リソース管理

- OpenGL コンテキストの適切な管理
- MIDI デバイスの初期化/終了処理

### インポート構造

- 相対インポートを統一的に使用
- `from engine.io import arc` パターンを推奨

### 座標系と描画サイズ

- 基本的に A4(297mm, 210mm)を想定している。何かを描画するときは、そこに収まるくらいの大きさにする。
- 扱う座標系は 3 次元のみ。vertices_list: list[np.ndarray(N, 3)]

## トラブルシューティング

### MIDI 関連

- デバイスが認識されない場合: `arc.list_midi_devices()`で確認
- MIDI 初期化エラー: `arc.start(midi=False)`で非 MIDI mode 実行

### レンダリング関連

- OpenGL エラー: グラフィックスドライバーの確認
- パフォーマンス問題: `engine/monitor/`のプロファイリング使用
