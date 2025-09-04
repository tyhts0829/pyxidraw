# PyxiDraw4 チュートリアル（2025-09 改訂）

新アーキテクチャ（Geometry 統一・関数エフェクト・E.pipeline）に沿って、最短経路で「作って動かす」体験を提供します。各ステップは独立実行でき、実運用に直結する API を優先して解説します。

## クイックスタート（最短 60 秒）

```python
# tutorials/00_quickstart.py
from api import E, G, run

def draw(t, cc):
    base = G.sphere(subdivisions=0.4).scale(90, 90, 90).translate(150, 150, 0)
    pipeline = (E.pipeline
                .ripple(amplitude=0.12, frequency=0.25)
                .rotate(angles_rad=(0.0, t*0.01, 0.0))
                .build())
    return pipeline(base)

if __name__ == "__main__":
    run(draw, canvas_size=(300, 300))  # MIDI不要
```

実行:

```bash
python tutorials/00_quickstart.py
```

ポイント:
- 形状は `G.*` で生成し、変換は `Geometry` の純メソッド（`.scale/.translate/.rotate`）。
- エフェクトは純関数（`Geometry -> Geometry`）を `E.pipeline` で構成し、`build()(g)` で適用。

## 6 本のチュートリアル

それぞれ 3–10 分で到達可能。順番に進めるのが最短です。

### 01. 基本形状と座標系（`01_basic_shapes.py`）
- `G.sphere()` など基本形状の生成
- `Geometry.scale/translate/rotate` による変換
- スケッチ関数 `draw(t, cc) -> Geometry` と `run(...)`

学べること: 位置（mm）、回転（ラジアン）、スケール（倍率）の扱い

### 02. 形状の合成（`02_multiple_shapes.py`）
- `G.empty()` と `+` 演算子で合成
- 複数形状のレイアウト手法

学べること: 合成の心構え（各形状は独立に純変換 → 合成）

### 03. 関数エフェクト入門（`03_basic_effects.py`）
- `E.pipeline` でエフェクトチェーンを構築
- 例: `displace(amplitude_mm=...)` と `rotate(angles_rad=...)`
- 時間 `t` を使ったアニメーション

学べること: パイプラインのビルドと適用、パラメータの変化設計

### 04. カスタム形状（`04_custom_shapes.py`）
- `@shape` + `BaseShape` で新しい形状を登録
- `Geometry.from_lines` による低レベル構築

学べること: 形状のアルゴリズム設計と API への組み込み

### 05. カスタムエフェクト（`05_custom_effects.py`）
- `@effects.registry.effect` で関数エフェクトを登録
- 例: `gradient(...)`（デモとして z を微小偏移）

学べること: パラメータ検証（`__param_meta__`）とレジストリ運用

### 06. 応用パイプライン（`06_advanced_pipeline.py`）
- 条件分岐・LOD・簡易キャッシュの設計
- 複数オブジェクトの一括管理
- MIDI 連携はオプショナル（デフォルト有効）

学べること: 実運用を想定した組み立て・負荷制御

### 07. MIDI 入門（`07_midi_basics.py`）
- 依存導入とポート列挙（`--list-ports`）
- `config.yaml` の `midi_devices` 設定（`port_name`/`cc_map`）
- `run(..., use_midi=True)` と `draw(t, cc)` の CC 受け取り
- CC 値（0..1）で `E.pipeline` のパラメータを制御

学べること: 実機/シミュレーション双方での CC 連動パターン

## 実行方法とヒント

```bash
# 任意のチュートリアルを実行（MIDIは既定で有効）
python tutorials/01_basic_shapes.py

# ヘッドレス検証（ウィンドウなしで配列サイズのみ確認）
PYXIDRAW_HEADLESS=1 python tutorials/02_multiple_shapes.py

# MIDI 入門の実行（シミュレーションで可視化）
python tutorials/07_midi_basics.py

# MIDI 入門の実行（実機制御）
PYXIDRAW_USE_MIDI=0 python tutorials/07_midi_basics.py  # 環境変数で無効化も可能

# 利用可能なポートを一覧
python tutorials/07_midi_basics.py --list-ports
```

- 既定のキャンバスは mm 単位。例: `run(..., canvas_size=(300, 300))`。
- MIDI は既定で有効。使わない場合は `run(..., use_midi=False)` または `python main.py --no-midi` を使用。
- 厳格モード（未接続/未導入時に終了）: `python main.py --midi-strict` または `PYXIDRAW_MIDI_STRICT=1`。
- 既定の有効化/厳格挙動は `config.yaml` の `midi.enabled_default` / `midi.strict_default` でも切替可能。

## 破壊的変更の要点（2025-09-03）

- Geometry 統一: 変換は `Geometry.scale/translate/rotate/concat`（純関数）
- エフェクト関数化: `@effects.registry.effect` で登録する `Geometry -> Geometry` のみ
- パイプライン一本化: `E.pipeline ... .build()(g)`（単層キャッシュ）
- 代表的な置換: `size→scale`, `at→translate`, `spin→rotate(0..1→2π)`, `move→translate`, `grow→scale`

## パイプライン仕様のシリアライズ/検証（実務向け）

```python
from api import E, to_spec, from_spec, validate_spec

pipeline = (E.pipeline
             .rotate(angles_rad=(0.5 * 3.141592653589793, 0.0, 0.0))
             .displace(amplitude_mm=0.2)
             .build())

spec = to_spec(pipeline)      # List[{"name": str, "params": dict}]
validate_spec(spec)           # 未登録名/不正パラメータを早期失敗
pipeline2 = from_spec(spec)   # 検証済み spec から Pipeline を復元
```

## トラブルシューティング（最新仕様）

- ImportError: プロジェクトルートから実行（`python tutorials/...`）。
- 依存関係: 最小構成は `pip install numpy numba pyglet moderngl psutil pyyaml`。
- レンダリング: OpenGL 実装（GPU ドライバ/仮想環境）の更新を確認。
- MIDI: 使わない場合は `use_midi=False` または `--no-midi`。既定は有効。ポート名は `config.yaml` を確認。

## 次の一歩

- `benchmarks/` で性能を測定（`python -m benchmarks run`）。
- `docs/guides/effects_authoring.md` を読み、独自エフェクトを追加。
- `tests/` のパターンに倣い、重要ユーティリティに単体テストを追加。

## 参考

- 設計方針と開発規約: `AGENTS.md`
- サンプルコード: `main.py`, `simple.py`
