# アーキテクチャ概要（Purpose & Architecture）

## 目的（What & Why）
- プロシージャルな線描（ラインベース）の幾何を生成・加工・描画するための軽量フレームワーク。
- 形状（Shapes）→ エフェクト（Effects）→ レンダリング（Engine）の責務分離により、再利用性とテスト容易性を確保。
- ライブ操作（MIDI/時間`t`）でパラメータを変調し、リアルタイムに結果を確認できる。

## まず押さえる前提（Units/座標系）
- 単位はミリメートル[mm]が既定。`util.constants.CANVAS_SIZES` の定義を基にキャンバスの実寸を決める。
- ウィンドウ解像度は `canvas_size(mm) × render_scale(px/mm)`。つまり 1mm は `render_scale` ピクセルに相当。
- 座標系はスクリーン座標: 原点はキャンバス左上、+X 右、+Y 下、Z は奥行き（深度テストは既定で未使用）。
  - ライン描画は正射影。`api.runner` で ModernGL の射影行列を設定しており、mm→NDC 変換を一意に定義。

## 中核コンセプト
- Geometry（統一表現）
  - `coords: float32 (N,3)` と `offsets: int32 (M+1,)` によるポリライン集合の正規化表現。
  - すべての変換は純関数（新しい Geometry を返す）。`digest: bytes` により内容指紋を保持（キャッシュ鍵）。
  - 不変条件: `offsets[0]==0` かつ `offsets[-1]==len(coords)`。i 本目の線は `coords[offsets[i]:offsets[i+1]]`。
  - 2D 入力は Z=0 で補完し常に (N,3) へ正規化。空集合は `coords.shape==(0,3)`, `offsets=[0]`。
- ファクトリとレジストリ
  - Shapes: `shapes/` + `@shape` で登録、`G.<name>(...) -> Geometry` を提供。
  - Effects: `effects/` + `@effect` で登録、`E.pipeline.<name>(...)` でチェーン可能。
  - 正規化キー（Camel→snake, lower）で一貫性を担保（`common/base_registry.py`）。
  - Effects はオプションで `__param_meta__` を公開でき、`validate_spec` が型/範囲/choices を追加検証。
- パイプライン
  - `PipelineBuilder` でステップを組み立て、`build()` で `Pipeline` を生成。
  - 厳格モード（既定 `strict=True`）で未知パラメータを検出。`to_spec/from_spec/validate_spec` でシリアライズと検証。
  - LRU 風の単層キャッシュ（インスタンス内）: 入力 `Geometry.digest` × パイプライン定義ハッシュでヒット判定。
    - 既定サイズは無制限。`.cache(maxsize=0)` で無効化、`.cache(maxsize=N)` で上限設定。
    - 既定値は環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` でも上書き可能。
  - ジオメトリ側の `digest` は環境変数 `PXD_DISABLE_GEOMETRY_DIGEST=1` で無効化可能（パイプラインは配列から都度ハッシュでフォールバック）。

## データフロー（概略）
```
G.<shape>() --> Geometry --(E.pipeline.*.build())--> Pipeline(Geometry->Geometry)
       \
        +---> translate/scale/rotate (Geometry API) ------------------+

user draw(t, cc) -> Geometry  --WorkerPool--> SwapBuffer --Renderer(ModernGL)--> Window(HUD)
                                                   ^                        |
                                              StreamReceiver <--- FrameClock + Tickables
```

### 実行ループと並行性（Frame/Tick モデル）
- `FrameClock` が登録された Tickable を固定順序で毎フレーム実行。
- `WorkerPool`（multiprocessing）が `draw(t, cc)` を非同期に実行して `Geometry` を生成。
- `StreamReceiver` は結果キューを読み、最新フレームのみを `SwapBuffer` に反映（古いフレームは棄却）。
- `LineRenderer` は `SwapBuffer` の front を検知して GPU に頂点群を一括転送し、`LINE_STRIP + primitive restart` で描画。
- 例外はワーカ側で `WorkerTaskError` に包んでメインスレッドに再送出（デバッグ容易性と失敗の早期顕在化）。

## 主なモジュール
- `api/`
  - `__init__.py`: 公開面（`G`, `E`, `run`/`run_sketch`, `Geometry`）。
  - `pipeline.py`: `Pipeline`, `PipelineBuilder`, 仕様検証（`validate_spec`）。
  - `shape_factory.py`: 形状ファクトリ。
  - `runner.py`: 実行エンジンの束ね（ModernGL/Pyglet、MIDI、ワーカー、HUD）。
- `engine/`
  - `core/geometry.py`: 統一 `Geometry`、基本変換、`digest`。
  - `core/frame_clock.py`, `core/tickable.py`: フレーム調停と更新インターフェース。
  - `pipeline/`: `WorkerPool`, `StreamReceiver`, `buffer` 等の並行処理。
  - `render/renderer.py`: ライン描画（正射影行列、倍精度→GPU転送）。
  - `ui/overlay.py`, `monitor/`: HUD とメトリクス。
  - `io/`: MIDI 接続・スナップショット取得。
- `effects/`: 幾何処理のオペレータ群と `registry.py`。
- `shapes/`: プリミティブ形状と `registry.py`。
- `common/`, `util/`: ロギング、型、幾何ユーティリティ、定数、設定ロード。

## API 境界と依存方向
- 外部利用者は `from api import G, E, run, Geometry` のみを前提にする。
- 依存は下向き（api → engine/common/util/effects/shapes）。`engine` は `api` を参照しない。
- JSON ライクな `spec` でパイプラインを保存/復元でき、ワークフローの再現性を確保。

### パイプライン spec の例
```json
[
  {"name": "rotate", "params": {"pivot": [200,200,0], "angles_rad": [0,0,0.5]}},
  {"name": "displace", "params": {"amplitude_mm": 0.5, "spatial_freq": 0.01, "t_sec": 1.0}}
]
```
`Pipeline.to_spec()`/`from_spec()` で相互変換でき、`validate_spec()` が未知キーや値の妥当性を検査する。

## 実行と拡張の最小例
```python
# 形状生成 → パイプライン → 実行（main.py の簡略版）
from api import G, E, run

def draw(t, cc):
    # t は秒、cc は MIDI CC（0-127 の Mapping）。
    # 時間/CC 依存パラメータがある場合は毎フレーム Pipeline を構築する（キャッシュはほぼミスする前提）。
    g = G.sphere(subdivisions=cc[1])
    pipe = (
        E.pipeline
         .rotate(angles_rad=(cc[3], cc[4], cc[5]), pivot=(200, 200, 0))
         .displace(amplitude_mm=cc[6]*50, spatial_freq=(0.01,0.01,0.01), t_sec=t)
         .build()
    )
    return pipe(g.scale(400*cc[8]).translate(200, 200, 0))

if __name__ == "__main__":
    run(draw, canvas_size=(400,400), render_scale=4, use_midi=True)
```

Tips:
- パラメータが不変の Pipeline は 1 度だけ構築して再利用すると高速（`.cache()` も有効活用）。
- 逆に、`t` や CC に依存する Pipeline は毎フレーム再構築でも OK（インスタンス内キャッシュは使われないためメモリ増は抑制される）。

## 非目標（Out of Scope）
- 高機能なDCC/ノードエディタUIや重厚な3Dレンダラは対象外（軽量リアルタイム線描に特化）。
- 互換性維持よりもシンプルさと純関数性を優先（Deprecated API は段階的廃止）。

## MIDI と入力（要点）
- `run(..., use_midi=True)` で可能なら実機 MIDI に接続。未接続時はフォールバック（`midi_strict=True` で失敗を致命扱い）。
- `draw(t, cc)` の `cc` は `Mapping[int, int]`（0–127）。`engine/io` がスナップショットを供給。

## テストとの接点（要点）
- `Geometry` はスナップショット（`digest`）で回帰検知しやすい。
- パイプラインは `strict` と `validate_spec` により境界が明確。キャッシュは `cache(maxsize=...)` で制御可能。

## 拡張のガイド（最短ルート）
- Shape の追加: `shapes/` に実装し `@shape` で登録。`generate(**params) -> Geometry` を返す。
- Effect の追加: `effects/` に `def effect_name(g: Geometry, *, ...) -> Geometry` を実装し `@effect` を付与。
  - 可能なら `__param_meta__ = {"param": {"type": "number", "min": 0, ...}}` を添えて `validate_spec` を強化。

既存のコーディング規約やテスト方針はリポジトリの「Repository Guidelines」を参照。
