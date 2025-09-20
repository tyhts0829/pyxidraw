# コードレビュー（src/ ディレクトリ別）

対象: `src/` 配下の主要ディレクトリ（API/Common/Effects/Engine*/Shapes/Util）
観点: 設計整合・型/ドキュメント・規約準拠（AGENTS.md）・安全性・拡張性・依存性・パフォーマンス

---

## api/

- 良い点
  - 薄い公開層が明確。`api/__init__.py` にエクスポート集約、`G`/`E` が使いやすい。
  - `effects.Pipeline` のキャッシュ鍵（ジオメトリ指紋 + パイプライン指紋）が堅牢。`validate_spec()` も厳格で安全。
  - 0..1 正規化を `engine.ui.parameters` の解決層で扱い、関数本体は実レンジで純粋関数化する設計が明快。
  - `shapes.ShapesAPI` の LRU による再生成抑制、`__getattr__` 動的解決が簡潔。

- 懸念/改善余地
  - ドキュメント上、「ユーザ入力は 0..1 正規化、関数受取は実レンジ」の二層構造がやや伝わりにくい。公開 API の docstring に明記したい。
  - `PipelineBuilder` 既定 `cache_maxsize=None`（無制限）は意図通りだが、メモリ増加時の指針を docstring に補足すると親切。

- 提案
  - `api/effects.py: EffectsAPI`/`PipelineBuilder` に、正規化→実レンジの流れとキャッシュ運用の注意を NumPy 形式で簡潔に追記。
  - `api/sketch.py` は丁寧な長文ドキュメントだが、先頭に「最小実行手順（数行）」の Quick Start を入れると導入が速い。

参考: `src/api/effects.py`, `src/api/shapes.py`, `src/api/sketch.py`

---

## common/

- 良い点
  - `param_utils.py` のハッシュ可能化/安定化はキャッシュ鍵として十分堅牢。`ensure_vec3` など小粒ユーティリティも適切。
  - `base_registry.py` のキー正規化（Camel→snake、lower、`-`→`_`）が統一され、shapes/effects で対称運用。

- 懸念/改善余地
  - 例外メッセージは十分だが、`_normalize_key` の ValueError/TypeError を public API 側 docstring にも明記すると利用者が把握しやすい。

- 提案
  - `param_utils.py` に 0..1 正規化ヘルパ群の使用例を 2–3 行追加（何をいつ使うかの指針）。

参考: `src/common/base_registry.py`, `src/common/param_utils.py`, `src/common/types.py`

---

## effects/

- 良い点
  - すべて関数ベース・純粋関数で一貫。`__param_meta__` が整備され、UI/検証層との連携が良い。
  - 幾何変換は `Geometry` に委譲し、ここではパラメトリック処理に専念（責務分離が明快）。
  - 代表例: `translate/scale/rotate` のメタ/型/docstring が模範的。`displace/wobble/ripple/fill` はパフォーマンス配慮あり。

- 懸念/改善余地
  - 重依存の直 import:
    - `numba`（例: `effects/fill.py`, `effects/displace.py`）
    - `shapely`（`effects/offset.py`）
    いずれも未導入環境で import 時点で失敗する。実行パスに入ったときのみ遅延 import し、未導入時は低速フォールバックにするのが安全（Ask-first 指針とも整合）。
  - 正規化入力の扱い: 関数は実レンジ受取で設計されているが、docstring には「UI/ランタイム経由では 0..1 で指定する。直呼び時は実レンジ」と明記したほうが規約理解が進む。

- 提案
  - 依存の遅延 import + フォールバック方針を統一（例: `try: import numba as nb; njit = nb.njit; except ImportError: def njit(*a, **k): return (lambda f: f)`）。
  - `__param_meta__` に `step` が入るものは可能な範囲で指定（UI スライダの粒度最適化）。
  - `effects/shader` 相当は存在せず、GL は render 側に隔離済みで OK。

参考: `src/effects/*.py`, 特に `src/effects/fill.py`, `src/effects/displace.py`, `src/effects/offset.py`

---

## engine/core/

- 良い点
  - `Geometry` の不変条件・表現統一・digest 設計が明快。`as_arrays(copy=False)` の読み取り専用ビューは安全で賢い。
  - 基本変換（translate/scale/rotate/concat）を最小提供に抑え、責務が明確。
  - `RenderWindow` は MSAA 設定や draw コールバック登録が簡潔で扱いやすい。

- 懸念/改善余地
  - `RenderWindow` の MSAA `Config` が環境により失敗する場合がある。フォールバック（samples=0）を docstring か実装で補助すると堅牢。
  - `geometry.digest`: 無効化時の RuntimeError は仕様通りだが、`api/effects` のフォールバックルートへの参照を docstring 冒頭に短く再掲すると親切。

- 提案
  - `render_window.py` に簡易フォールバック Config（例: try/except → 再生成）をコメントで示すか、オプション引数で許可。

参考: `src/engine/core/geometry.py`, `src/engine/core/render_window.py`, `src/engine/core/frame_clock.py`

---

## engine/io/

- 良い点
  - `manager.connect_midi_controllers()` が遅延 import を用い、未導入環境への配慮あり。例外方針が分かりやすい。
  - `controller.MidiController` は 7bit/14bit の正規化、永続化（JSON）を持ち、実用十分。

- 懸念/改善余地
  - `controller.py` はトップレベルで `import mido` しているため、未導入環境での import 失敗があり得る。`manager` 同様の遅延 import が無難。
  - Mac 固有名（Intech Grid）への同期コードは残しつつ、存在確認とログだけに留めており安全だが、他環境では無効である旨のコメントを補うと親切。

- 提案
  - `controller.py` も遅延 import に統一し、`api/sketch.py` 側の strict/非 strict 方針と整合させる。

参考: `src/engine/io/manager.py`, `src/engine/io/controller.py`

---

## engine/render/

- 良い点
  - `LineRenderer` に描画責務集約。`LineMesh` が VBO/IBO/VAO を一括管理し、Primitive Restart でライン群を効率描画。
  - シェーダは最小構成で可読性が高い。投影行列は上位（API 側）で設定済み。

- 懸念/改善余地
  - `shader.py` にモジュールヘッダがない（他ファイルは概ねある）。短いモジュール docstring を付与したい。
  - ライン太さ/色は Uniform 固定。後段の Parameter GUI と繋ぐ拡張余地あり（必須ではない）。

- 提案
  - `shader.py` 冒頭に「どこで・何を・なぜ」を 3–4 行記述。将来的に可変ライン太さ/色のエントリポイントをコメントで案内。

参考: `src/engine/render/renderer.py`, `src/engine/render/line_mesh.py`, `src/engine/render/shader.py`

---

## engine/runtime/

- 良い点
  - `SwapBuffer` + `WorkerPool` + `StreamReceiver` の結線がシンプル。`WorkerTaskError` で文脈付き例外を親に伝播する設計が丁寧。
  - `WorkerPool` の `inline`（num_workers<1）モードは依存のない軽量実行に便利。

- 懸念/改善余地
  - `multiprocessing` 併用時、例外の型は message ベースで再構築される前提。必要なら `args` に構造化情報（frame_id 等）を追加する拡張余地あり。

- 提案
  - 監視ログの粒度（DEBUG/INFO）を `WorkerPool`/`StreamReceiver` で揃える小調整（任意）。

参考: `src/engine/runtime/buffer.py`, `src/engine/runtime/worker.py`, `src/engine/runtime/receiver.py`

---

## engine/ui/parameters/

- 良い点
  - `ParameterRuntime`/`ValueResolver`/`RangeHint` による「正規化（0..1）↔実レンジ」変換が体系化されており、提案ルールと合致。
  - 既定値・メタ無し時のフォールバックや、ベクトル値のコンポーネント分解など、実用的な細部が詰められている。

- 懸念/改善余地
  - `value_resolver.py` は大きい。関数群は整理されているが、README 的に「流れ図（merge → resolve scalar/vector → register → denormalize）」を 1 段書くと理解しやすい。

- 提案
  - `Parameters` サブパッケージの Overview を `engine/ui/parameters/AGENTS.md` に短く足す（役割対応表）。

参考: `src/engine/ui/parameters/*.py`

---

## shapes/

- 良い点
  - すべて関数ベース、`__param_meta__` 充実。ランダム性のある `asemic_glyph` は `random_seed` 受取で再現性担保（規約準拠）。
  - 重い幾何はキャッシュ（`lru_cache`）や分割生成でパフォーマンス配慮。

- 懸念/改善余地
  - `text.py` は macOS 固定のフォント探索経路を優先。Linux/Windows の一般的パスも候補に入れるか、`fontconfig` 連携をコメントで誘導すると移植性が上がる。
  - `grid.py` の `__param_meta__` は `type: "number"` で tuple を許容しているが、UI 側ベクトル表現と整合が取れているか（現状は動作OK）。必要なら `type: "number"` + vec 要素の min/max/step の明記を追加検討。

- 提案
  - 各 shape の docstring 冒頭に「入力は 0..1（UI 正規化）、直呼びは実レンジ」の一文を追記（effects と揃える）。
  - `text.py` のフォント探索に OS 判別と Warning を追加（現状でも Warning は一部あり）。

参考: `src/shapes/*.py`, 特に `src/shapes/text.py`, `src/shapes/sphere.py`, `src/shapes/grid.py`

---

## util/

- 良い点
  - `constants.py` のキャンバス定義や Perlin 定数は集中管理で明確。`utils.load_config()` はフェイルソフトで安全。
  - 3D 変換ユーティリティ `geom3d_ops.py` は `njit` 最適化と数値安定性に配慮。

- 懸念/改善余地
  - `geom3d_ops.py` も `numba` 直 import。未導入環境での import 失敗を避けるには遅延/フォールバックが望ましい。

- 提案
  - `utils._find_project_root()` の探索条件に `pyproject.toml`/`setup.cfg` を既に含めており十分。コメントに探索順の例を 1 行追記するとより親切。

参考: `src/util/constants.py`, `src/util/utils.py`, `src/util/geom3d_ops.py`

---

## 横断的な所見（全体）

- 強み
  - 設計の分離（Shapes/Geometry/Effects/Render/Runtime）が明快で、拡張しやすい。
  - 0..1 正規化を UI/ランタイムで扱い、関数群は実レンジで純粋関数に統一するアーキテクチャは可読・保守に優れる。
  - 型注釈・docstring（日本語）が行き届き、例外メッセージも有益。

- リスク/改善優先度（高→中）
  1) 重依存（numba/shapely/fontTools/mido）の直 import による import 時失敗
     - 対策: 各モジュールで遅延 import + フォールバック（低速実装 or 明示的 RuntimeError）を統一。
  2) 正規化入力と実レンジの関係が初見で少し伝わりにくい
     - 対策: 公開層 docstring（`api/*`, `shapes/*`, `effects/*`）に 1 行ルールを追記。
  3) 一部ファイルでモジュールヘッダ欠如（例: `engine/render/shader.py`）
     - 対策: 先頭ヘッダ（どこで・何を・なぜ）を短く追加。

- 小さな提案
  - `__param_meta__` に `step` を積極設定（UI/自動テストの安定化）。
  - `RenderWindow` の MSAA Config 失敗時フォールバック（サンプル数 0）をコメントまたは実装に反映。

---

以上。

