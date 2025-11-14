# アーキテクチャ概要（Purpose & Architecture）

## アーキテクチャキャンバス（1-pager）

- 層（数値は内外の序数。小さいほど内側）
  - L0 Core/Base: `common/`, `util/`, `engine/core/`
  - L1 Domain/Transforms: `shapes/`, `effects/`（純関数 `Geometry -> Geometry`）
  - L2 Infra & Runtime: `engine/render/`, `engine/runtime/`, `engine/ui/`, `engine/io/`, `engine/export/`
  - L3 API/Entry: `api/`, `main.py`
- 許可する依存方向
  - 外側 → 内側（同層は許可）。数値で表すと「ソース層 ≥ ターゲット層」。
  - 例: L3→L2/L1/L0, L2→L1/L0（ただし下記の個別禁止エッジに従う）。
- 不変条件（抜粋）
  - 循環依存を作らない。
  - `print()` を禁止（ログは `logging` 経由）。
  - `common/` と `util/` は上位層（`api/`, `engine/*`, `effects/`, `shapes/`）に依存しない。

### 個別禁止エッジ（本リポ特化の契約）

- `engine/*` → `api/*` を禁止（エンジン層は公開 API を知らない）。
- `engine/*` → `effects/*`, `shapes/*` を禁止（設計の一方向性維持）。
- `effects/*`, `shapes/*` → `engine/render/*`, `engine/runtime/*`, `engine/ui/*`, `engine/io/*` を禁止。
- `engine/runtime/*` が `effects/*` の関数を直接呼ぶことを禁止（パイプライン適用は `api` の責務）。
- `effects.registry`/`shapes.registry` の参照は `api/*`, `effects/*`, `shapes/*` に限定。
  - `engine/*`, `common/*`, `util/*` からの参照は禁止（登録/解決の境界を越えない）。
- 例外が必要な場合は ADR を追加し、ここ（キャンバス）にも例外行を追記すること。

## 目的（What & Why）

- プロシージャルな線描（ラインベース）の幾何を生成・加工・描画するための軽量フレームワーク。
- 形状（Shapes）→ エフェクト（Effects）→ レンダリング（Engine）の責務分離により、再利用性とテスト容易性を確保。
- ライブ操作（MIDI/時間`t`）でパラメータを変調し、リアルタイムに結果を確認できる。

## まず押さえる前提（Units/座標系）

- 単位はミリメートル[mm]が既定。`util.constants.CANVAS_SIZES` の定義を基にキャンバスの実寸を決める。
- ウィンドウ解像度は `canvas_size(mm) × render_scale(px/mm)`。つまり 1mm は `render_scale` ピクセルに相当。
- 座標系はスクリーン座標: 原点はキャンバス左上、+X 右、+Y 下、Z は奥行き（深度テストは既定で未使用）。
  - ライン描画は正射影。`api.sketch` で ModernGL の射影行列を設定しており（行列の構築は `api.sketch_runner.utils.build_projection`）、mm→NDC 変換を一意に定義。

## 中核コンセプト

- Geometry（統一表現）
  - `coords: float32 (N,3)` と `offsets: int32 (M+1,)` によるポリライン集合の正規化表現。
  - すべての変換は純関数（新しい Geometry を返す）。キャッシュは「Lazy 署名（`api.lazy_signature`）」を基準に行う（Geometry 自体は digest を持たない）。
  - 不変条件: `offsets[0]==0` かつ `offsets[-1]==len(coords)`。i 本目の線は `coords[offsets[i]:offsets[i+1]]`。
  - 2D 入力は Z=0 で補完し常に (N,3) へ正規化。空集合は `coords.shape==(0,3)`, `offsets=[0]`。
  - `as_arrays(copy=False)` は読み取り専用ビューを返す（キャッシュ整合性を維持）。
- 書き込みが必要な場合は `copy=True` を使用する。
- LazyGeometry（遅延 spec）
  - `base`: shape 実装（関数参照）とパラメータ、または実体 `Geometry`。
  - `plan`: effect 実装（関数参照）とパラメータの列。順序は固定。
  - `realize()`: `base` を実行して `Geometry` を得た後、`plan` を順次適用。engine.core は registries を参照せず、API 層が注入した関数参照のみを用いる。
- ファクトリとレジストリ
  - Shapes: `shapes/` + `@shape` で登録。`G.<name>(...)` は既定で `LazyGeometry`（spec）を返し、終端で `realize()` して `Geometry` を得る。
- Effects: `effects/` + `@effect` で登録。`E.pipeline.<name>(...)` でチェーンし、`LazyGeometry` の plan にエフェクト実装（関数参照）を積む。
  - 例: `mirror`（2D 放射状/直交の軽量版）、`mirror3d`（真の 3D 放射状: 球面くさび・大円境界）。

### Effects 概要

- パイプライン
  - `PipelineBuilder` でステップを組み立て、`build()` で `Pipeline` を生成。
  - 共通バイパス: 各エフェクトは共通の `bypass: bool` を持つ。Parameter GUI からのトグル、または `E.pipeline.<effect>(..., bypass=True)` 明示引数でスキップできる。
    - GUI 優先順位: 「明示引数 > GUI > 既定値」。
    - 実行: `bypass=True` のステップは Pipeline に追加されず、実行負荷は発生しない。
    - キャッシュ: `bypass` は署名生成に含めないため、キャッシュ鍵に影響しない。
  - 厳格検証（build 時の未知パラメータ検出）は行わない。実行時に関数シグネチャで自然に検出され得る。
  - 単層 LRU キャッシュ（インスタンス内）: キーは `(b"sig", lazy_signature_for(LazyGeometry))`。
    - 既定サイズは無制限。`.cache(maxsize=0)` で無効化、`.cache(maxsize=N)` で上限設定。
    - 既定値は環境変数 `PXD_COMPILED_CACHE_MAXSIZE` で上書き可（実装準拠）。
    - 実装は `OrderedDict` による LRU 風。
  - Lazy 署名（`api.lazy_signature`）は shape/plan ごとに「関数 ID（`module:qualname`／`common.func_id.impl_id`）」「量子化後パラメータ署名（`common.param_utils.params_signature`）」を積み、128bit に集約。
  - 中間結果（prefix 単位）の LRU は `LazyGeometry.realize()`（engine.core）側にあり、最長一致の prefix を再利用する。
- パラメータ GUI（Dear PyGui）
  - `engine.ui.parameters` パッケージ（`ParameterRuntime`, `FunctionIntrospector`, `ParameterValueResolver`, `ParameterStore`, `ParameterWindow` 等）が shape/effect 引数を検出し、Dear PyGui による独立ウィンドウで表示/編集する（実体は `engine.ui.parameters.dpg_window`）。
  - `ParameterRuntime` は `FunctionIntrospector`/`ParameterValueResolver` を介してメタ情報抽出と Descriptor 登録を行い、GUI override を適用してから元の関数へ委譲（変換レイヤは廃止し、実値を扱う）。
  - RangeHint は実レンジ（min/max/step）のヒントのみを提供する。UI は表示比率を計算してクランプするが、内部値はクランプしない。
  - GUI 有効時は `engine.ui.parameters.manager.ParameterManager` が `user_draw` をラップし、初回フレームで自動スキャン →`ParameterWindowController` を起動。
  - 外観設定は `util.utils.load_config()` で読み込む `parameter_gui` キー（`configs/default.yaml` / ルート `config.yaml`）から解決し、`ParameterWindowController` → `ParameterWindow` に渡す（ウィンドウ寸法/タイトル、スタイル/色）。設定未指定時は既定の最小テーマで動作。
  - カテゴリ別のヘッダ色（Display/HUD/shape/pipeline）は `parameter_gui.theme.categories` で個別指定可能。未指定時は `theme.colors.header*` を使用。
  - カテゴリ名の決定規則: effect パラメータは `pipeline_label`（`.label(uid)` で指定）≫ `pipeline_uid`（例: `p0`）≫ `scope` の優先で決定。表示ラベルは内部 UID と分離され、Parameter ID は `pipeline_uid` に基づくため衝突しない。
  - パラメータ GUI はメインスレッドで維持しつつ、ワーカ側へは GUI 値のスナップショットを渡して適用する（SnapshotRuntime）。このため GUI 有効時でも `WorkerPool` は並列実行できる。
  - 駆動方式は内部ドライバで抽象化。可能なら `pyglet.clock.schedule_interval` に統合（メインスレッドから `render_dearpygui_frame()` を実行）、未導入時はバックグラウンドスレッドで `start_dearpygui()` を実行。Dear PyGui 未導入環境では GUI を起動しない限り import は行われない（スタブは用意していないため、起動時は未導入で ImportError となる）。

## データフロー（概略）

```
G.<shape>() --> Geometry --(E.pipeline.*.build())--> Pipeline(Geometry->Geometry)
       \
        +---> translate/scale/rotate (Geometry API) ------------------+

user draw(t) -> Geometry  --WorkerPool--> SwapBuffer --Renderer(ModernGL)--> Window(HUD)
                                                   ^                        |
                                              StreamReceiver <--- FrameClock + Tickables
```

### 実行ループと並行性（Frame/Tick モデル）

- `FrameClock` が登録された Tickable を固定順序で毎フレーム実行。
- `WorkerPool`（既定は multiprocessing。Parameter GUI 有効時もスナップショット注入により並列実行が可能）が `draw(t)` を実行して `Geometry` を生成。
- `StreamReceiver` は結果キューを読み、最新フレームのみを `SwapBuffer` に反映（古いフレームは棄却）。
- `LineRenderer` は `SwapBuffer` の front を検知して GPU に頂点群を一括転送し、`LINE_STRIP + primitive restart` で描画。
- 例外はワーカ側で `WorkerTaskError` に包んでメインスレッドに再送出（デバッグ容易性と失敗の早期顕在化）。

## 主なモジュール

- `api/`
  - `__init__.py`: 公開面（`G`, `E`, `run`/`run_sketch`, `Geometry`）。
- `effects.py`: `Pipeline`, `PipelineBuilder`。
  - `shapes.py`: 形状 API。
  - `sketch.py`: 実行エンジンの束ね（ModernGL/Pyglet、MIDI、ワーカー、HUD）。初期化・補助は内部ヘルパ `api/sketch_runner/*.py` に委譲。
- `engine/`
  - `core/geometry.py`: 統一 `Geometry`、基本変換。
  - `core/frame_clock.py`, `core/tickable.py`: フレーム調停と更新インターフェース。
  - `runtime/`: `WorkerPool`, `StreamReceiver`, `buffer` 等の並行処理。
  - `render/renderer.py`: ライン描画（正射影行列、倍精度 →GPU 転送）。
  - `ui/hud/overlay.py`, `ui/hud/sampler.py`: HUD とメトリクス。
  - `ui/hud/`: `HUDConfig` とフィールド定義（HUD 表示/計測のオプション制御）。
  - キャッシュ累計スナップショット取得は `api.sketch_runner.utils.hud_metrics_snapshot`（トップレベル関数）で実装し、Worker へ関数注入する（engine は api を参照しない）。
  - `io/`: MIDI 接続・スナップショット取得。
- `effects/`: 幾何処理のオペレータ群と `registry.py`。
  - 代表例: `affine(auto_center: bool, pivot: Vec3, angles_rad: Vec3, scale: Vec3, delta: Vec3)` —
    `auto_center=True` でジオメトリの平均座標を中心に使用。`False` で `pivot` を使用。
- `shapes/`: プリミティブ形状と `registry.py`。
- `common/`, `util/`: ロギング、型、幾何ユーティリティ、定数、設定ロード。

## API 境界と依存方向

- 外部利用者は `from api import G, E, run, Geometry` のみを前提にする。
- 依存は下向き（api → engine/common/util/effects/shapes）。`engine` は `api` を参照しない。
- 外部保存/復元/仕様検証の API は提供しない（縮減方針）。

## 実行と拡張の最小例

```python
# 形状生成 → パイプライン → 実行（main.py の簡略版）
from api import E, G, run


def draw(t):
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
    run(
        draw,
        canvas_size=(400, 400),
        render_scale=4,  # float 可（例: 2.5）
        use_midi=True,
        # クリップ空間基準の線幅（既定 0.0006）。mm 指定は将来拡張予定。
        line_thickness=0.0006,
        # 線色（RGBA 0–1 またはヘックス文字列）。RGB 長さ3も許容（α=1.0 補完）。
        line_color="#000000",
    )
```

Tips:

- パラメータが不変の Pipeline は 1 度だけ構築して再利用すると高速（`.cache()` も有効活用）。
- 逆に、`t` や CC に依存する Pipeline は毎フレーム再構築でも OK（インスタンス内キャッシュは使われないためメモリ増は抑制される）。

## 非目標（Out of Scope）

- 高機能な DCC/ノードエディタ UI や重厚な 3D レンダラは対象外（軽量リアルタイム線描に特化）。
- 互換性維持よりもシンプルさと純関数性を優先（Deprecated API は段階的廃止）。

## MIDI と入力（要点）

- `run(..., use_midi=True)` で可能なら実機 MIDI に接続。未接続時は警告の上でフォールバック。
- 線の太さは `run(..., line_thickness=0.0006)` で指定可能（クリップ空間 -1..1 基準の半幅相当）。
- 線の色は `run(..., line_color=(r,g,b,a) | "#RRGGBB[AA]")` で指定。RGB（3 要素）の場合は α=1.0 を補完。
- 背景色 `background` も RGBA タプルまたはヘックスで指定可能。
- `background` / `line_color` を未指定の場合は、`configs/default.yaml` の
  `canvas.background_color` / `canvas.line_color` をフォールバックとして使用。
- Parameter GUI の保存値がある場合は、起動直後に Store→ 描画へ初期適用（背景は `RenderWindow.set_background_color`、線色は `LineRenderer.set_line_color`）。
  優先順位は「引数 > 保存値 > config」。
- HUD の色（`runner.hud_text_color`、`runner.hud_meter_color`、`runner.hud_meter_bg_color`）も起動直後に Store→Overlay へ初期適用する（`OverlayHUD.set_text_color` / `set_meter_color` / `set_meter_bg_color`）。

フォント探索/適用:

- 設定（`configs/default.yaml` → `config.yaml` 上書き）で `fonts.search_dirs` を指定すると、Text シェイプ（`src/shapes/text.py:108`）のフォント探索リストに最優先で追加される。拡張子は `.ttf/.otf/.ttc`。
- HUD（`src/engine/ui/hud/overlay.py`）は起動時に `fonts.search_dirs` 配下から必要なフォントのみを `pyglet.font.add_file()` で登録する（`hud.load_all_fonts=false` 既定、family 名に部分一致）。`hud.font_name` / `hud.font_size` を用いて描画フォントを決定する。HUD のフォント指定は family 名を想定する。
- Parameter GUI（`src/engine/ui/parameters/dpg_window.py`）は Dear PyGui 起動時に `fonts.search_dirs` からフォントファイルを列挙し、`parameter_gui.layout.font_name` の部分一致候補（優先: `.ttf`）を順次 `dpg.add_font()` でトライし、最初に成功したものを `dpg.bind_font()` で適用する（失敗時は Dear PyGui 既定フォント）。Parameter GUI の指定はファイル名ベース（family 名ではなくファイルパス登録）である。
  優先順位は「保存値 > config」。HUD 表示の有効/無効は `run(..., show_hud=...)` で指定可能。
  `show_hud=None` かつ Parameter GUI 有効時は、GUI の「Show HUD（runner.show_hud）」で表示/非表示を動的に切替できる（明示引数がある場合は引数優先でロック）。
- CC は引数で渡さない。`from api import cc` で `cc[i]` を参照（MIDI の 0.0–1.0）。`WorkerPool` が各フレームのスナップショットを供給。
  - Engine/UI 側は `util.cc_provider.get_cc_snapshot()` を経由して CC スナップショットを参照し、`engine/* -> api/*` の依存を避ける（登録は `api.cc` が `set_cc_snapshot_provider` で行う）。

## テストとの接点（要点）

- `Geometry` は `coords/offsets` の JSON スナップショットで回帰検知しやすい（digest は廃止）。
- パイプラインは build 時検証を行わず、実行時に自然に検出される。キャッシュは `cache(maxsize=...)` で制御可能。

## 拡張のガイド（最短ルート）

- Shape の追加: `shapes/` に実装し `@shape` で登録。`generate(**params) -> Geometry` を返す。
- Effect の追加: `effects/` に `def effect_name(g: Geometry, *, ...) -> Geometry` を実装し `@effect` を付与。
  - 可能なら `__param_meta__ = {"param": {"type": "number", "min": 0, ...}}` を添えて、UI 表示のヒント（RangeHint 構築）を提供。

既存のコーディング規約やテスト方針はリポジトリの「Repository Guidelines」を参照。

---

## Export（保存系コンポーネントの要点）

- 画像（PNG）: `engine/export/image.py`
  - 低コスト: ウィンドウのカラーバッファをそのまま保存（HUD 含む）。
  - 高品質: FBO にラインのみを描画して保存（HUD なし、スケール対応）。
- G-code: `engine/export/service.py` + `engine/export/gcode.py`
  - 非ブロッキングに変換・保存。HUD へ進捗通知。
- 動画（MP4）: `engine/export/video.py`
  - 最小の同期レコーダ。`imageio`/`imageio-ffmpeg` が存在すれば、1 フレームずつ取りこぼしなく書き出す。
  - HUD 含む録画（画面バッファ）/HUD なし録画（FBO + `LineRenderer.draw()`）を選択可能。

## 座標変換と投影（詳細）

- 物理単位は mm。ウィンドウ解像度は `canvas_size(mm) × render_scale(px/mm)`（render_scale は float 可）。
- 投影は正射影（擬似 2D）。Y 軸はスクリーン座標に合わせて上が負になる変換を適用。
  - `api.sketch.run_sketch()` が ModernGL 用の 4x4 行列（`api.sketch_runner.utils.build_projection`）を構築し、`engine.render.renderer.LineRenderer` に渡す。
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
  - コンストラクタで dtype/形状を検証し、`coords` は float32 C-contig、`offsets` は int32 C-contig へ強制正規化。
- 署名（キャッシュ協調）
  - Geometry 自体に digest は保持しない。パイプラインのキャッシュは `api.lazy_signature.lazy_signature_for(LazyGeometry)` により生成される 128bit 署名を用いる。
  - 署名は base（実体 Geometry は `id()`、shape は `impl_id + params_signature`）と plan（各 `impl_id + params_signature`）を順に積んで `blake2b-128` に集約。

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

### レイヤー描画とフォールバック（on_draw 先行対策）

- 背景
  - `style` は非幾何エフェクトとして plan に積まれるが、ワーカ側で抽出し描画レイヤーへ変換する。
  - 抽出後、幾何の plan から `style` ステップは除去され、幾何計算（キャッシュ）に影響しない。
- 生成→受信→描画の流れ
  - Worker: `engine/runtime/worker.py` が `draw(t)` の戻り値を正規化し、`StyledLayer[]` に展開（`style` の後勝ちを反映）。
  - Receiver: `engine/runtime/receiver.py` が最新フレームのみを取り出し、`layers`（または `geometry`）を `SwapBuffer.push(...)` する。
  - Renderer（tick）: `engine/render/renderer.py` が `SwapBuffer.try_swap()` に成功したら front を取り出し、`layers` のときは `_frame_layers` に保持、`geometry` のときは即アップロード。
  - Renderer（draw）: `_frame_layers` があれば、各レイヤーの `color/thickness` を適用してアップロード→描画を順次実行。
- フォールバック再描画（本件の安定化）
  - 背景などの GUI 操作により `on_draw` が `tick` より先行して発火すると、レイヤー未到達のフレームが挟まることがある。
  - その場合でも見た目が崩れないよう、`renderer.draw()` は直前フレームで描いたレイヤーのスナップショット（`StyledLayer` + 実体 `Geometry`）を保持し、`_frame_layers` が無いフレームではそれを再描画する。
  - これにより「片方のレイヤーだけが描かれる」「単色化する」などの点滅/消失を回避する。
  - 実装: `engine/render/renderer.py`（`_last_layers_snapshot` の保持と再描画）。


## Optional Dependencies（方針）

- 実行ポリシー
  - 性能目的の遅延のみ採用。重たい依存は「使う関数/メソッド内」でローカル import する。
  - トップレベルの try-import/sentinel/専用例外は用いない（ImportError はそのまま上げる）。
  - 依存未導入時の挙動は各レイヤの責務で扱う（例: API 層で Null フォールバック、または機能をスキップ）。
- 代表モジュールの現状
  - ModernGL: 描画パスの必須依存。`engine.render.renderer` および `api.sketch_runner.render` などでトップレベル import。ヘッドレス検証のみ行う場合は `run_sketch(..., init_only=True)` で GL 初期化を回避可能。
  - pyglet: 使用箇所でローカル import。`export.image` と Parameter GUI は実行時にコントローラ経由で読み込む（`dpg_window` は Dear PyGui をトップレベルで import するが、GUI を開始したときのみモジュールが import される）。
  - mido: `engine.io.manager`/`engine.io.controller` ともにローカル import。未導入時は API 層がフォールバック（MIDI 無効）。
  - shapely: `effects.offset/partition` は処理関数内でローカル import。未導入時は `partition` が耳切り三角分割にフォールバック。
  - numba: デコレータ `njit` は未導入時 no-op で吸収（例: `effects.dash`）。
  - fontTools/fontPens: `shapes.text` は `get_font`/`get_glyph_commands` 内のローカル import。

## 並行処理（WorkerPool / StreamReceiver / SwapBuffer）

- 登場要素
  - `WorkerPool`: `multiprocessing.Process` を N 個起動し、`draw(t)` をバックグラウンド実行。
  - `StreamReceiver`: 結果キューをポーリングし、最新フレームのみを `SwapBuffer` に保存。
  - `SwapBuffer`: front/back のダブルバッファと `version` カウンタ、イベントで構成。
- データの流れ
  1. `FrameClock.tick()` ごとに `WorkerPool.tick()` が `RenderTask(frame_id, t, cc_state)` を `task_q` へ投入。
  2. 各ワーカは API 層から注入された関数で `api.cc.set_snapshot(cc_state)` を呼び、ついでに `draw(t)` を実行 →`RenderPacket(geometry, frame_id)` を `result_q` へ。
  3. `StreamReceiver.tick()` が `result_q` を非ブロッキングで最大 K 件（既定 2）処理し、最新 `frame_id` のみを `SwapBuffer.push()`。
  4. `LineRenderer.tick()` が `SwapBuffer.try_swap()` を呼び、準備済みなら front/back を交換。
- バックプレッシャ/スキップ
  - `WorkerPool` は `task_q` の `maxsize=2*num_workers` で自然な抑制。詰まれば新タスク投入をスキップ。
  - `StreamReceiver` は「最新フレーム以外を捨てる」戦略で遅延伝播を防止。
- 終了処理
  - `WorkerPool.close()` は `None` センチネルを投入 →`join(timeout)`→ 生存時 `terminate()`。
  - 例外はワーカ側で `WorkerTaskError(frame_id, original)` に包んでメインへ再送出。

## パイプライン（キャッシュ/厳格検証の詳細）

- キャッシュ鍵
  - `Pipeline` の LRU は `(b"sig", api.lazy_signature.lazy_signature_for(LazyGeometry))` をキーにする。
  - 形状結果 LRU と中間（prefix）LRU は `engine.core.lazy_geometry.LazyGeometry.realize()` 側にあり、`impl_id` と量子化後パラメータ署名で構成。
- 容量/動作
  - `PipelineBuilder.cache(maxsize=None|0|N)`、または `PXD_PIPELINE_CACHE_MAXSIZE` で設定。
  - 実装は `OrderedDict` による LRU 風（ヒットで末尾へ、上限超過で先頭を追い出し）。
  - 中間キャッシュ（prefix 単位）: 各ステップ適用後の `Geometry` を `LazyGeometry` 側のグローバル LRU に保存。
    - 鍵は `(base_key, prefix_effect_sigs[:i])`（`impl_id` と量子化後パラメータ署名の列）。
    - 環境変数で制御: `PXD_PREFIX_CACHE_ENABLED`, `PXD_PREFIX_CACHE_MAXSIZE`, `PXD_PREFIX_CACHE_MAX_VERTS`。
    - 終端ミス時に最長一致の prefix を再利用し、残りのステップのみ実行する。
- 厳格検証
  - build 時に未知キーは検証しない（厳格モードは廃止）。
    （削除）

## レジストリと公開 API

- Shapes（`shapes/registry.py`）
  - `@shape`/`@shape()`/`@shape("name")` で登録。キーは Camel→snake 小文字化で正規化。
  - 形状は関数として `@shape` で登録し、戻り値は `Geometry` または `Geometry.from_lines()` 可能な
    「ポリライン列（list/ndarray の列）」とする。旧形式 `(coords, offsets)` は非サポート。
  - 高水準 API `G` は `ShapesAPI` のインスタンス。`G.polygon(...) -> Geometry` のように関数的に呼び出す。
- Effects（`effects/registry.py`）
  - `@effect` で `def effect_name(g: Geometry, *, ...) -> Geometry` な関数を登録。
  - パラメータメタ `__param_meta__`（任意）は UI 表示のヒントとして利用する。
- パラメータ規約（重要）
  - すべての Shape/Effect 関数の引数は None を受け付けない（Optional 禁止）。既定値にも None を使用しない。
  - 数値系（float/int/bool/vector）は実値で受け取る。
  - RangeHint（表示レンジ）は `__param_meta__` の min/max/step を参照。
  - キャッシュ鍵（署名）生成時は `__param_meta__['step']` を用いて「float のみ量子化」する（int/bool は非量子化）。未指定時の既定は 1e-6（`PXD_PIPELINE_QUANT_STEP` で上書き可）。ベクトルは成分ごとに適用し、step の成分が不足する場合は末尾値で補完する。
  - 備考: Effects は量子化後の値がそのまま実行引数になる。Shapes はキャッシュ鍵に量子化を使うが、実行にはランタイム解決後の値（非量子化）を渡す。
  - 非数値は GUI スライダー対象外。文字列は自由入力（既定は単一行）。
    - 複数行にしたい場合は `__param_meta__` に `{type: "string", multiline: true, height: 80}` のように指定する（`height` は任意）。
    - 列挙（choices あり）は選択肢（ラジオ/コンボ）として扱う。
  - 根拠: `engine.ui.parameters` のスライダーは None を扱わず、表示時に `float(None)` が例外となるため。
- パラメータ UI/cc のルール（現行仕様）
  - cc は `api.cc` のグローバル辞書（`cc[i] -> float(0..1)`、未定義は 0.0）。`draw(t)` 内から数値式として自由に利用する。
  - GUI は「draw 内で未指定（＝シグネチャの既定値が採用された）引数のみ」を対象として表示・調整する。
  - 値の優先順位は「明示引数 > GUI > 既定値」。MIDI による GUI 上書きは行わない（midi_override は廃止）。
  - RangeHint は `__param_meta__` がある場合のみ使用し、無い場合は 0–1 の既定レンジで扱う（クランプは表示上のみ）。
  - Runtime は `set_inputs(t)` のみ受け取り、cc は関与しない（cc は `api.cc` でフレーム毎に更新される）。
- 公開面
- 利用者は `from api import G, E, run, Geometry` のみに依存。
  - 上位（api）→ 下位（engine/common/util/effects/shapes）と一方向の依存。`engine` は `api` を知らない。
  - 破壊的変更はスタブ同期テストが検出する。意図的な場合はスタブ再生成手順に従う。

## スタブ生成と CI（型の同期）

- 目的
  - `api/__init__.pyi` に「利用者が見る API 形状」を自動生成し、実装と同期を保つ。
- 更新手順
  - `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`
- CI/テスト
  - `.github/workflows/verify-stubs.yml` が `tests/test_g_stub_sync.py`, `tests/test_pipeline_stub_sync.py` を実行し、スタブ・実装の不整合を検知。

## 設計ガードレール（検証）

- アーキテクチャテスト
  - `tests/test_architecture.py`（導入時）で import グラフを静的解析し、数値レイヤ規則（外側 → 内側のみ）と上記「個別禁止エッジ」を検証する。
  - 本リポはトップレベルで `api|engine|effects|shapes|common|util` の 6 種ディレクトリを起点に層付けする。
- スタブ同期
  - 公開 API の形状はスタブ生成＋同期テストで検証（破壊的変更の早期検知）。
- ヘルスチェック（ローカル運用の目安）
  - 変更後に `ruff`/`mypy`/`pytest` の最小セットを回す。死蔵コードの検知には `vulture` を随時使用。

## 設定と環境変数

- 設定ファイル
  - `util.utils.load_config()` が `configs/default.yaml` → ルート `config.yaml` を順に読み込み（トップレベルのみ上書き）。
  - `runner.run_sketch()` で `fps` や MIDI 既定（`midi.strict_default`）の補完に利用。
- 主な環境変数
  - `PXD_DISABLE_GEOMETRY_DIGEST=1` … Geometry のダイジェスト計算を無効化（ベンチ比較等）。
  - `PXD_PIPELINE_CACHE_MAXSIZE=<int>` … パイプライン単層キャッシュの上限。`0` で無効、未設定は無制限。

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
  from common.types import Vec3
  from engine.core.geometry import Geometry

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
  def star(*, points: int = 5, r_outer: float = 50, r_inner: float = 20) -> Geometry:
      th = np.linspace(0, 2 * np.pi, points * 2, endpoint=False)
      rr = np.where(np.arange(points * 2) % 2 == 0, r_outer, r_inner)
      xy = np.c_[rr * np.cos(th), rr * np.sin(th)]
      return Geometry.from_lines([xy])
  ```

- 追加後の手順
  - `effects/registry.py`/`shapes/registry.py` は自動登録済み（デコレータ）。
- スタブ更新: `python -m tools.gen_g_stubs`。
  - テスト: `pytest -q -m smoke` で簡易確認 → `pytest -q`。

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
- `ShapesAPI` の LRU は CPython 実装のロックで基本安全。`generate()` は純関数であること。
- `SwapBuffer` はロック/イベントでスレッドセーフ。`try_swap()` は非ブロッキング。

## 既知の制限/非目標（補足）

- 3D の隠面消去や奥行バッファは非対応。線描に特化。
- ハイ DPI の正確な mm→px 換算は OS/ディスプレイ設定に依存（`render_scale` で実用上の見た目を調整）。
- ワーカは Python マルチプロセスのため、起動コストや共有メモリの制約がある。短時間のスケッチでは `workers=1` も検討。

## 参考: 主要モジュールの対応表

- API: `src/api/__init__.py`, `effects.py`, `sketch.py`, `shapes.py`, `lfo.py`
- API (internal helpers): `src/api/sketch_runner/*.py`（`utils.py`, `midi.py`, `render.py`, `export.py`, `params.py`, `recording.py`）
- Engine/Core: `core/geometry.py`, `core/frame_clock.py`, `core/render_window.py`, `core/tickable.py`
- Engine/Runtime: `runtime/worker.py`, `runtime/receiver.py`, `runtime/buffer.py`
- Engine/Render: `render/renderer.py`, `render/line_mesh.py`, `render/shader.py`
- Engine/UI/Monitor: `ui/hud/overlay.py`, `ui/hud/sampler.py`
- Effects: `effects/*.py`, `effects/registry.py`
- Shapes: `shapes/*.py`, `shapes/registry.py`
- Common/Util: `common/*.py`, `util/*.py`（`constants.py`, `utils.py`, `geom3d_ops.py` など）

---

## ADR（設計決定メモ）

- 例外やルール変更は `docs/adr/` にミニ ADR（10 行程度）を追加し、背景/決定/代替/影響を残す。
- 例外追加時は「アーキテクチャキャンバス」の個別禁止エッジにも例外を 1 行追記して整合を保つ。

---

このドキュメントは実装（src/ 配下）と同期しています。差分を見つけた場合は、該当コードの参照箇所とともに更新してください。
