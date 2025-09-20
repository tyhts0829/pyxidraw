# コードレビュー（src/ ディレクトリ別）

対象: `src/` 配下の主要ディレクトリ（API/Common/Effects/Engine\*/Shapes/Util）
観点: 設計整合・型/ドキュメント・規約準拠（AGENTS.md）・安全性・拡張性・依存性・パフォーマンス

---

## api/

<!--
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

— 簡素化/削減の観点 —
- `PipelineBuilder.strict` 既定 True は堅牢だが、開発初期の反復時に冗長な失敗を誘発することがある。既定を False にして必要時にオン、もしくは `PXD_STRICT_DEFAULT` 環境変数制御にすると体験が軽くなる（互換性要検討）。
- `to_spec/from_spec` はメソッドの薄いラッパ。API 面の明快さ優先なら維持、縮減方針なら片方（関数 or メソッド）へ集約可能。
- `__repr__` の詳細構築は開発用利便。パフォーマンス影響は小さく、削減優先度は低。 -->

---

## common/

<!--
- 良い点

  - `param_utils.py` のハッシュ可能化/安定化はキャッシュ鍵として十分堅牢。`ensure_vec3` など小粒ユーティリティも適切。
  - `base_registry.py` のキー正規化（Camel→snake、lower、`-`→`_`）が統一され、shapes/effects で対称運用。

- 懸念/改善余地

  - 例外メッセージは十分だが、`_normalize_key` の ValueError/TypeError を public API 側 docstring にも明記すると利用者が把握しやすい。

- 提案
  - `param_utils.py` に 0..1 正規化ヘルパ群の使用例を 2–3 行追加（何をいつ使うかの指針）。

参考: `src/common/base_registry.py`, `src/common/param_utils.py`, `src/common/types.py`

— 簡素化/削減の観点 —

- `BaseRegistry._normalize_key` の厳格チェック（空文字/型）は妥当。削るとデバッグ難度が上がるため現状維持推奨。
- `param_utils.make_hashable_param` はケース網羅が広いが、実運用で不要な分岐（numpy object 配列対応等）を絞れば数行削減可。ただしキャッシュ安定性に影響するため慎重。 -->

---

## effects/

<!-- - 良い点

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

— 簡素化/削減の観点 —

- デッドコード: `effects/offset.py` の `_determine_join_style()` は未使用。安全に削除可能。
- 直 import 依存: `numba`/`shapely` は関数使用直前に遅延 import へ変更可能（失敗時は純 numpy の低速版や明示的な RuntimeError を出す）。import 時失敗を避けつつコード量の重複を減らせる。
- `fill.py` の Numba 最適化群は維持方針で妥当だが、軽量化優先なら line/cross/dots いずれかを簡易版へ統合し、バッチ関数（`generate_line_intersections_batch` など）を共通化して行数削減が可能。
- `displace.py` の Perlin 実装は自前で詳細だが、改善の余地は「fade/lerp/grad の docstring 簡略化」「Permutation の 2 倍連結生成を関数化（3–5 行削減）」程度。性能と可読性のバランスは現状良好。
- `ripple/wobble` は frequency 正規化ロジックが類似。小ユーティリティ化（タプル化ヘルパ）で重複削減可。 -->

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

— 簡素化/削減の観点 —

- `geometry.digest` の環境変数制御とフォールバック説明はやや長め。実装はシンプルで、削る必要性は低。docstring を 1–2 文圧縮の余地はあるが、可読性を損なうほどではない。
- `render_window.py` は MSAA 設定が一発で決まらない環境があるため、`Config` の try/except を入れるだけで運用トラブルが減る。コード量増は最小で実効性高。

---

## engine/io/

<!-- - 良い点

  - `manager.connect_midi_controllers()` が遅延 import を用い、未導入環境への配慮あり。例外方針が分かりやすい。
  - `controller.MidiController` は 7bit/14bit の正規化、永続化（JSON）を持ち、実用十分。

- 懸念/改善余地

  - `controller.py` はトップレベルで `import mido` しているため、未導入環境での import 失敗があり得る。`manager` 同様の遅延 import が無難。
  - Mac 固有名（Intech Grid）への同期コードは残しつつ、存在確認とログだけに留めており安全だが、他環境では無効である旨のコメントを補うと親切。

- 提案
  - `controller.py` も遅延 import に統一し、`api/sketch.py` 側の strict/非 strict 方針と整合させる。

参考: `src/engine/io/manager.py`, `src/engine/io/controller.py`

— 簡素化/削減の観点 —

- 直 import（`mido`）を遅延化。`manager.py` と同様のガードにより import 時失敗を回避しつつ、例外は「使用時」に限定。
- `MidiController` 内の `process_7bit_control_change`/`process_14bit_control_change`/`calc_combined_value` は構造化が良い。行数削減のみ目的なら 7bit/14bit 分岐を 1 関数に合流できるが、可読性低下の恐れあり。現状維持推奨。
- JSON 永続化の `by_name` 以外の構造は未使用。将来的な拡張予定が無ければキーを固定化してローダの分岐を簡略化可（数行削減）。 -->

---

## engine/render/

<!-- - 良い点

  - `LineRenderer` に描画責務集約。`LineMesh` が VBO/IBO/VAO を一括管理し、Primitive Restart でライン群を効率描画。
  - シェーダは最小構成で可読性が高い。投影行列は上位（API 側）で設定済み。

- 懸念/改善余地

  - `shader.py` にモジュールヘッダがない（他ファイルは概ねある）。短いモジュール docstring を付与したい。
  - ライン太さ/色は Uniform 固定。後段の Parameter GUI と繋ぐ拡張余地あり（必須ではない）。

- 提案
  - `shader.py` 冒頭に「どこで・何を・なぜ」を 3–4 行記述。将来的に可変ライン太さ/色のエントリポイントをコメントで案内。

参考: `src/engine/render/renderer.py`, `src/engine/render/line_mesh.py`, `src/engine/render/shader.py`

— 簡素化/削減の観点 —

- 現行は Geometry Shader でライン太さを実装。プラットフォーム互換性を最大化するなら GS 無し + `LINE_STRIP` + 固定幅に落とす選択肢もある（コードは若干減る）。ただし見た目品質/太さ可変性が落ちるため要トレードオフ検討。
- `renderer._geometry_to_vertices_indices` は明快。微小削減案として `np.add`/`np.arange` の一括生成と `indices[::line+1]=restart` のようなベクトル化があるが、可読性とバグリスクを考えると現状維持で良い。 -->

---

## engine/runtime/

<!-- - 良い点

  - `SwapBuffer` + `WorkerPool` + `StreamReceiver` の結線がシンプル。`WorkerTaskError` で文脈付き例外を親に伝播する設計が丁寧。
  - `WorkerPool` の `inline`（num_workers<1）モードは依存のない軽量実行に便利。

- 懸念/改善余地

  - `multiprocessing` 併用時、例外の型は message ベースで再構築される前提。必要なら `args` に構造化情報（frame_id 等）を追加する拡張余地あり。

- 提案
  - 監視ログの粒度（DEBUG/INFO）を `WorkerPool`/`StreamReceiver` で揃える小調整（任意）。

参考: `src/engine/runtime/buffer.py`, `src/engine/runtime/worker.py`, `src/engine/runtime/receiver.py`

— 簡素化/削減の観点 —

- `WorkerTaskError.__reduce__` によるメッセージ再構築は堅牢。簡素化優先なら、例外オブジェクト自体を送らず `(frame_id, error_str)` のタプルに統一し、受信側で例外化する方式にできる（数行削減・pickle 安定）。
- `WorkerPool.inline` 分岐は価値が高く、行数は最小限。現状維持推奨。 -->

---

## engine/ui/parameters/

<!-- - 良い点

  - `ParameterRuntime`/`ValueResolver`/`RangeHint` による「正規化（0..1）↔ 実レンジ」変換が体系化されており、提案ルールと合致。
  - 既定値・メタ無し時のフォールバックや、ベクトル値のコンポーネント分解など、実用的な細部が詰められている。

- 懸念/改善余地

  - `value_resolver.py` は大きい。関数群は整理されているが、README 的に「流れ図（merge → resolve scalar/vector → register → denormalize）」を 1 段書くと理解しやすい。

- 提案
  - `Parameters` サブパッケージの Overview を `engine/ui/parameters/AGENTS.md` に短く足す（役割対応表）。

参考: `src/engine/ui/parameters/*.py`

— 簡素化/削減の観点 —

- 大きいのは `value_resolver.py`。ロジックは整理されているため大規模削減は非推奨だが、`_range_hint_from_meta` 近傍のデフォルト/推測処理を 2–3 個の小関数へ分割して見通しを改善可能（コード行数は横ばい、可読性向上）。
- UI 未対応型（enum/vector 以外）の分岐は明示的で読みやすい。削減対象ではない。 -->

---

## shapes/

- 良い点

  - すべて関数ベース、`__param_meta__` 充実。ランダム性のある `asemic_glyph` は `random_seed` 受取で再現性担保（規約準拠）。
  - 重い幾何はキャッシュ（`lru_cache`）や分割生成でパフォーマンス配慮。

- 懸念/改善余地

  - `text.py` は macOS 固定のフォント探索経路を優先。Linux/Windows の一般的パスも候補に入れるか、`fontconfig` 連携をコメントで誘導すると移植性が上がる。
  - `grid.py` の `__param_meta__` は `type: "number"` で tuple を許容しているが、UI 側ベクトル表現と整合が取れているか（現状は動作 OK）。必要なら `type: "number"` + vec 要素の min/max/step の明記を追加検討。

- 提案
  - 各 shape の docstring 冒頭に「入力は 0..1（UI 正規化）、直呼びは実レンジ」の一文を追記（effects と揃える）。
  - `text.py` のフォント探索に OS 判別と Warning を追加（現状でも Warning は一部あり）。

参考: `src/shapes/*.py`, 特に `src/shapes/text.py`, `src/shapes/sphere.py`, `src/shapes/grid.py`

— 簡素化/削減の観点 —

- デッドコード: `shapes/text.py` の `_process_vertices_batch_fast()` は未使用。削除可能。
- `text.py` の最適化関数群（njit）は価値があるが、1 ファイル当たりの行数が大きい。`TextRenderer` と njit 群をモジュール分割（`text_runtime.py` など）すると可読性が上がる（コード量は同等、見通し改善）。
- `sphere.py` は生成バリエーションが豊富。最小構成を狙うならスタイルを 3 種へ絞る（latlon/icosphere/rings）ことで導入負荷を軽減できるが、表現力低下のため任意。
- `grid.py` のベクトル化は既に良好。さらなる削減余地は小。

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

— 簡素化/削減の観点 —

- `constants.NOISE_CONST` の巨大リテラルは視認性を損ねる。起動時生成（固定 seed で permutation/grad3 を合成）に切り替えればファイル行数を大幅削減可。ただし「定数の再現性」をどこまで重視するかの方針決めが必要。
- `geom3d_ops.py` の `numba` は遅延 import + フォールバック化（effects と同方針）で import 失敗を回避しつつ依存を軽めに見せられる。

---

## 横断的な所見（全体）

- 強み

  - 設計の分離（Shapes/Geometry/Effects/Render/Runtime）が明快で、拡張しやすい。
  - 0..1 正規化を UI/ランタイムで扱い、関数群は実レンジで純粋関数に統一するアーキテクチャは可読・保守に優れる。
  - 型注釈・docstring（日本語）が行き届き、例外メッセージも有益。

- リスク/改善優先度（高 → 中）

  1. 重依存（numba/shapely/fontTools/mido）の直 import による import 時失敗
     - 対策: 各モジュールで遅延 import + フォールバック（低速実装 or 明示的 RuntimeError）を統一。
  2. 正規化入力と実レンジの関係が初見で少し伝わりにくい
     - 対策: 公開層 docstring（`api/*`, `shapes/*`, `effects/*`）に 1 行ルールを追記。
  3. 一部ファイルでモジュールヘッダ欠如（例: `engine/render/shader.py`）
     - 対策: 先頭ヘッダ（どこで・何を・なぜ）を短く追加。

- 小さな提案
  - `__param_meta__` に `step` を積極設定（UI/自動テストの安定化）。
  - `RenderWindow` の MSAA Config 失敗時フォールバック（サンプル数 0）をコメントまたは実装に反映。

---

以上。
