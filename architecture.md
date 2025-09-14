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
  - `as_arrays(copy=False)` は読み取り専用ビューを返す（digest/キャッシュ整合性を維持）。
    書き込みが必要な場合は `copy=True` を使用する。
- ファクトリとレジストリ
  - Shapes: `shapes/` + `@shape` で登録、`G.<name>(...) -> Geometry` を提供。
  - Effects: `effects/` + `@effect` で登録、`E.pipeline.<name>(...)` でチェーン可能。
  - 正規化キー（Camel→snake, lower, `-`→`_`）で一貫性を担保（`common/base_registry.py`）。
  - Effects はオプションで `__param_meta__` を公開でき、`validate_spec` が型/範囲/choices を追加検証。
- パイプライン
  - `PipelineBuilder` でステップを組み立て、`build()` で `Pipeline` を生成。
  - 厳格モード（既定 `strict=True`）で未知パラメータを検出。`to_spec/from_spec/validate_spec` でシリアライズと検証。
  - 単層 LRU キャッシュ（インスタンス内）: 入力 `Geometry.digest` × パイプライン定義ハッシュでヒット判定。
    - 既定サイズは無制限。`.cache(maxsize=0)` で無効化、`.cache(maxsize=N)` で上限設定。
    - 既定値は環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` でも上書き可能（負値は 0=無効 として扱う）。
    - 実装は `OrderedDict` による LRU 風で、get/set/evict は `RLock` で最小限保護（軽量なスレッド安全性）。
  - パイプライン定義ハッシュは、各ステップの「名前」「関数バイトコード近似（`__code__.co_code` の blake2b-64）」「正規化パラメータ（`common.param_utils.params_to_tuple`）の blake2b-64」を積み、128bit に集約。
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
    # t は秒、cc は MIDI CC の正規化値（Mapping[int, float], 0.0–1.0）。
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
- `draw(t, cc)` の `cc` は `Mapping[int, float]`（0.0–1.0 の正規化値）。`engine/io` がスナップショットを供給。

## テストとの接点（要点）
- `Geometry` はスナップショット（`digest`）で回帰検知しやすい。
- パイプラインは `strict` と `validate_spec` により境界が明確。キャッシュは `cache(maxsize=...)` で制御可能。

## 拡張のガイド（最短ルート）
- Shape の追加: `shapes/` に実装し `@shape` で登録。`generate(**params) -> Geometry` を返す。
- Effect の追加: `effects/` に `def effect_name(g: Geometry, *, ...) -> Geometry` を実装し `@effect` を付与。
  - 可能なら `__param_meta__ = {"param": {"type": "number", "min": 0, ...}}` を添えて `validate_spec` を強化。

既存のコーディング規約やテスト方針はリポジトリの「Repository Guidelines」を参照。

---

## 座標変換と投影（詳細）
- 物理単位は mm。ウィンドウ解像度は `canvas_size(mm) × render_scale(px/mm)`。
- 投影は正射影（擬似 2D）。Y 軸はスクリーン座標に合わせて上が負になる変換を適用。
  - `api.runner.run_sketch()` が ModernGL 用の 4x4 行列を構築し、`engine.render.renderer.LineRenderer` に渡す。
  - 実装行列（転置後にシェーダへ書き込み）:
    ```python
    proj = [[ 2/w,   0,   0, -1],
            [   0, -2/h,  0,  1],
            [   0,   0, -1,  0],
            [   0,   0,  0,  1]]
    ```
- Z はレイヤ表現（重なりの補助）。深度テストは使わず、`LINE_STRIP + primitive restart` で一括描画。

## Geometry（詳細仕様）
- メモリレイアウト
  - `coords: float32 (N,3)` 連続配列。`offsets: int32 (M+1,)` で M 本のポリライン境界を表す。
  - i 本目の線: `coords[offsets[i]:offsets[i+1]]`。
- 不変条件/規約
  - `offsets[0]==0`、`offsets[-1]==len(coords)`、2D 入力は `Z=0` 補完、空集合は `coords.shape==(0,3)`, `offsets=[0]`。
  - 変換は純関数（`translate/scale/rotate/concat`）。常に新インスタンスを返す。
- ダイジェスト（キャッシュ協調）
  - `digest: bytes` は `blake2b(digest_size=16)` による内容指紋。初回遅延計算→以後再利用。
  - 無効化: `PXD_DISABLE_GEOMETRY_DIGEST=1`。この場合 `g.digest` アクセスは例外、ただしパイプライン側が配列から都度ハッシュでフォールバック。

## レンダリング（ModernGL / 1 ドローの設計）
- 1 フレームの流れ
  - `LineRenderer.tick()` で新規フレームがあれば CPU 側で `coords/offsets` → `VBO/IBO` へ整形し GPU へアップロード。
  - `LineRenderer.draw()` で `mgl.LINE_STRIP` + プリミティブリスタートにより一括描画。
- IBO 生成規約
  - 各ポリラインのインデックス列の末尾に `PRIMITIVE_RESTART_INDEX (0xFFFFFFFF)` を挿入。
  - 総インデックス数は `len(coords) + 本数(M)`（各線に 1 つリスタートを追加）。
  - 実装: `engine.render.renderer._geometry_to_vertices_indices()`。
- ウィンドウ/GL
  - `engine.core.render_window.RenderWindow` が MSAA 有効（`samples=4`）で生成、初回描画時に画面中央へ配置。
  - `moderngl` にてブレンド有効化（`SRC_ALPHA, ONE_MINUS_SRC_ALPHA`）。

## 並行処理（WorkerPool / StreamReceiver / SwapBuffer）
- 登場要素
  - `WorkerPool`: `multiprocessing.Process` を N 個起動し、`draw(t, cc)` をバックグラウンド実行。
  - `StreamReceiver`: 結果キューをポーリングし、最新フレームのみを `SwapBuffer` に保存。
  - `SwapBuffer`: front/back のダブルバッファと `version` カウンタ、イベントで構成。
- データの流れ
  1) `FrameClock.tick()` ごとに `WorkerPool.tick()` が `RenderTask(frame_id, t, cc)` を `task_q` へ投入。
  2) 各ワーカが `draw()` 実行→`RenderPacket(geometry, frame_id)` を `result_q` へ。
  3) `StreamReceiver.tick()` が `result_q` を非ブロッキングで最大 K 件（既定 2）処理し、最新 `frame_id` のみを `SwapBuffer.push()`。
  4) `LineRenderer.tick()` が `SwapBuffer.try_swap()` を呼び、準備済みなら front/back を交換。
- バックプレッシャ/スキップ
  - `WorkerPool` は `task_q` の `maxsize=2*num_workers` で自然な抑制。詰まれば新タスク投入をスキップ。
  - `StreamReceiver` は「最新フレーム以外を捨てる」戦略で遅延伝播を防止。
- 終了処理
  - `WorkerPool.close()` は `None` センチネルを投入→`join(timeout)`→生存時 `terminate()`。
  - 例外はワーカ側で `WorkerTaskError(frame_id, original)` に包んでメインへ再送出。

## パイプライン（キャッシュ/厳格検証の詳細）
- キャッシュ鍵
  - 入力 `geometry_hash` × `pipeline_key`。
  - `geometry_hash`: 通常は `Geometry.digest`（無効時は配列から都度 blake2b-128）。
  - `pipeline_key`: ステップ列を順にハッシュして合成。
    - 各ステップで `name.encode()`、エフェクト関数の近似版 `__code__.co_code` ハッシュ（blake2b-64）、正規化パラメータ（dict/配列の決定的 repr）ハッシュ（blake2b-64）を積む。
- 容量/動作
  - `PipelineBuilder.cache(maxsize=None|0|N)`、または `PXD_PIPELINE_CACHE_MAXSIZE` で設定。
  - 実装は `OrderedDict` による LRU 風（ヒットで末尾へ、上限超過で先頭を追い出し）。
- 厳格検証
  - `PipelineBuilder.strict(True)`（既定）でビルド時に各エフェクト関数シグネチャと `params` のキーを照合。未知キーがあれば `TypeError`。
  - `validate_spec(spec)` は JSON 風値（数値/文字列/真偽/None、list/dict の入れ子）も検査。さらにエフェクトが `__param_meta__` を公開していれば `type/min/max/choices` を照合。

## レジストリと公開 API
- Shapes（`shapes/registry.py`）
  - `@shape`/`@shape()`/`@shape("name")` で登録。キーは Camel→snake 小文字化で正規化。
  - 形状は `BaseShape.generate(**params)` を実装し、戻り値は `Geometry` または
    `Geometry.from_lines()` 可能な「ポリライン列（list/ndarray の列）」とする。
    旧形式の `(coords, offsets)` タプルは非サポート（参照: `src/api/shape_factory.py`）。
  - 高水準 API `G` は `ShapeFactory` のインスタンス。`G.circle(...) -> Geometry` のように関数的に呼び出す。
- Effects（`effects/registry.py`）
  - `@effect` で `def effect_name(g: Geometry, *, ...) -> Geometry` な関数を登録。
  - パラメータメタ `__param_meta__`（任意）を公開すれば `validate_spec()` が型/範囲/選択肢を追加検証。
- 公開面
  - 利用者は `from api import G, E, run, Geometry, to_spec, from_spec, validate_spec` のみに依存。
  - 上位（api）→下位（engine/common/util/effects/shapes）と一方向の依存。`engine` は `api` を知らない。

## スタブ生成と CI（型の同期）
- 目的
  - `api/__init__.pyi` に「利用者が見る API 形状」を自動生成し、実装と同期を保つ。
- 更新手順
  - `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`
- CI/テスト
  - `.github/workflows/verify-stubs.yml` が `tests/test_g_stub_sync.py`, `tests/test_pipeline_stub_sync.py` を実行し、スタブ・実装の不整合を検知。

## 設定と環境変数
- 設定ファイル
  - `util.utils.load_config()` が `configs/default.yaml` → ルート `config.yaml` を順に読み込み（トップレベルのみ上書き）。
  - `runner.run_sketch()` で `fps` や MIDI 既定（`midi.strict_default`）の補完に利用。
- 主な環境変数
  - `PXD_DISABLE_GEOMETRY_DIGEST=1` … Geometry のダイジェスト計算を無効化（ベンチ比較等）。
  - `PXD_PIPELINE_CACHE_MAXSIZE=<int>` … パイプライン単層キャッシュの上限。`0` で無効、未設定は無制限。
  - `PYXIDRAW_MIDI_STRICT=1|true|on|yes` … MIDI 厳格モード。初期化失敗を致命扱い（`SystemExit(2)`）。

## エラーハンドリング/ロギング
- ワーカ例外は `WorkerTaskError(frame_id, original)` に包んでメインへ伝搬。`StreamReceiver.tick()` で受領次第再送出（早期顕在化）。
- MIDI 初期化エラー
  - 厳格モード時は例外ログ後に終了。非厳格時は警告ログを出し Null 実装にフォールバック。
- ログは標準 `logging`。必要に応じてハンドラ/レベルを上位アプリで設定。

## パフォーマンス指針
- 形状
  - 変換は `Geometry` 側でまとめて適用（多段の numpy コピーを最小化）。
  - 形状生成は `G` の LRU（maxsize=128）でヒットさせる。パラメータをハッシュ可能に正規化しているため安定。
- パイプライン
  - パラメータ不変のときは `Pipeline` を 1 回構築して再利用（キャッシュが効く）。
  - `t`/MIDI 依存で毎フレーム組み直す場合はキャッシュヒットを期待しない設計で OK（単層・無制限でも O(1) 近似）。
- レンダリング
  - 多数の線を「1 本の VBO/IBO + 1 ドロー」で描く設計。`primitive restart` によりドローコールを増やさない。
  - MSAA を有効化しつつ `render_scale` で視認性と速度のバランスを調整。
- 並行性
  - `workers` を CPU コア/処理負荷に合わせて調整。重いエフェクト（オフセット/塗り）ほどワーカ側で時間を使う。

## 拡張レシピ（スニペット）
- Effect を追加
  ```python
  # src/effects/wave.py
  from engine.core.geometry import Geometry
  from common.types import Vec3
  from .registry import effect

  @effect()
  def wave(g: Geometry, *, amplitude_mm: float = 1.0, spatial_freq: float = 0.02, t_sec: float = 0.0) -> Geometry:
      """Z に正弦波変位を加える簡易エフェクト。"""
      import numpy as np
      c, o = g.as_arrays(copy=False)
      if c.size == 0:
          return Geometry(c.copy(), o.copy())
      z = np.sin(c[:, 0] * spatial_freq + t_sec) * amplitude_mm
      out = c.copy(); out[:, 2] += z.astype(out.dtype)
      return Geometry(out, o.copy())

  # パラメータメタ（任意）
  wave.__param_meta__ = {
      "amplitude_mm": {"type": "number", "min": 0.0},
      "spatial_freq": {"type": "number", "min": 0.0},
      "t_sec": {"type": "number", "min": 0.0},
  }
  ```
- Shape を追加
  ```python
  # src/shapes/star.py
  import numpy as np
  from engine.core.geometry import Geometry
  from shapes.registry import shape

  @shape()
  class Star:
      def generate(self, *, points: int = 5, r_outer: float = 50, r_inner: float = 20) -> Geometry:
          th = np.linspace(0, 2*np.pi, points*2, endpoint=False)
          rr = np.where(np.arange(points*2) % 2 == 0, r_outer, r_inner)
          xy = np.c_[rr*np.cos(th), rr*np.sin(th)]
          return Geometry.from_lines([xy])
  ```
- 追加後の手順
  - `effects/registry.py`/`shapes/registry.py` は自動登録済み（デコレータ）。
  - スタブ更新: `python -m scripts.gen_g_stubs`。
  - テスト: `pytest -q -m smoke` で簡易確認→ `pytest -q`。

## シリアライズ仕様（詳細）
- 形式
  - `PipelineSpec = list[ {"name": str, "params": dict[str, JSONLike]} ]`
  - `JSONLike = int|float|str|bool|None|list[JSONLike]|dict[str,JSONLike]`
- 妥当性
  - `name` は登録済みエフェクトであること。
  - `params` は辞書（未知キーはエラー）。
  - メタがある場合は `type/min/max/choices` の範囲チェックを実施。
  - numpy 配列は不可（リストに変換して渡す）。

## スレッド/プロセス安全性
- `Pipeline` の内部キャッシュはインスタンスローカル。複数スレッドから共有する場合は外部ロックで保護するか、スレッドごとに別インスタンスを使う。
- `ShapeFactory` の LRU は CPython 実装のロックで基本安全。`generate()` は純関数であること。
- `SwapBuffer` はロック/イベントでスレッドセーフ。`try_swap()` は非ブロッキング。

## 既知の制限/非目標（補足）
- 3D の隠面消去や奥行バッファは非対応。線描に特化。
- ハイ DPI の正確な mm→px 換算は OS/ディスプレイ設定に依存（`render_scale` で実用上の見た目を調整）。
- ワーカは Python マルチプロセスのため、起動コストや共有メモリの制約がある。短時間のスケッチでは `workers=1` も検討。

## 参考: 主要モジュールの対応表
- API: `src/api/__init__.py`, `pipeline.py`, `runner.py`, `shape_factory.py`
- Engine/Core: `core/geometry.py`, `core/frame_clock.py`, `core/render_window.py`, `core/tickable.py`
- Engine/Pipeline: `pipeline/worker.py`, `pipeline/receiver.py`, `pipeline/buffer.py`
- Engine/Render: `render/renderer.py`, `render/line_mesh.py`, `render/shader.py`
- Engine/UI/Monitor: `ui/overlay.py`, `monitor/sampler.py`
- Effects: `effects/*.py`, `effects/registry.py`
- Shapes: `shapes/*.py`, `shapes/registry.py`
- Common/Util: `common/*.py`, `util/*.py`（`constants.py`, `utils.py`, `geom3d_ops.py` など）

---

このドキュメントは実装（src/ 配下）と同期しています。差分を見つけた場合は、該当コードの参照箇所とともに更新してください。
