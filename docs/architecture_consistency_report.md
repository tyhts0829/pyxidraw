# architecture.md と実装の整合性チェックレポート（自動生成）

生成日: 2025-10-07

本レポートは、リポジトリの `architecture.md` に記載された仕様と、`src/` 配下の実装との差異を機械的に点検した結果の要約です。主に依存方向（禁止エッジ）、ジオメトリ/レンダリング仕様、パイプライン/キャッシュ仕様、パラメータGUI仕様の観点で確認しました。

## 結論（サマリ）
- 依存方向・禁止エッジ・レンダリングフロー・MIDI/Runtime の大枠は実装と整合しています。
- 重要な不一致が2点、ドキュメンテーション上の軽微な不一致が1点見つかりました（下記詳細）。

## 確認条件と方法
- 依存/禁止エッジ違反の有無を `rg` によるインポート走査で確認。
- ジオメトリ/レンダリング/パイプライン/GUI 周辺の主要ファイルを直接精読し、`architecture.md` の当該箇所と突き合わせ。

対象ファイル例:
- API: `src/api/__init__.py`, `src/api/shapes.py`, `src/api/effects.py`, `src/api/sketch.py`
- Engine/Core/Render/Runtime/UI: `src/engine/core/*`, `src/engine/render/*`, `src/engine/runtime/*`, `src/engine/ui/*`
- Registry/Utils: `src/shapes/registry.py`, `src/effects/registry.py`, `src/common/param_utils.py`, `src/common/base_registry.py`

## 一致している主な点
- 依存方向と禁止エッジ
  - `engine/*` から `api/*`/`effects/*`/`shapes/*` への参照は見当たりませんでした（OK）。
  - `effects/*`/`shapes/*` は `engine/core` のみ参照し、`engine/{render,runtime,ui,io}` への参照は無し（OK）。
- プロジェクション/レンダリング
  - 正射影行列・ブレンド設定・MSAA・PRIMITIVE_RESTART は記載通り。
    - 行列定義: `src/api/sketch.py:292`
    - インデックス付与: `src/engine/render/renderer.py:121`
- パイプライン/キャッシュ方針
  - `Geometry.digest × pipeline_key` の鍵構成、LRU 風の `OrderedDict` 実装、builder の `.cache(maxsize)` と環境変数による制御は整合。
- GUI/パラメータ
  - 既定値で露出された引数のみ GUI 表示、RangeHint は `__param_meta__` がある場合に使用、既定レンジは 0–1、クランプは表示上のみ（OK）。

## 不一致（重要）

1) Shapes の「量子化値の扱い」
- 仕様（architecture.md）: 「Effects は量子化後の値を実行引数に使用。Shapes は鍵のみ量子化（実行はランタイム解決値＝非量子化）」と明記。
  - 該当: `architecture.md:249`
- 実装: `G.<name>(...)` 実行時、キャッシュ鍵生成に用いた「量子化後の値」をそのまま関数引数として渡しています。
  - 呼び出し経路: `src/api/shapes.py:113`（`params_tuple` を `dict(...)` に戻して `_generate_shape_resolved` に渡す）
- 影響: 量子化ステップ（既定 1e-6、環境で上書き可）により、Shape 関数に渡る実引数が微小に丸められます。設計意図（鍵のみ量子化）と異なるため、精度要求の高い Shape 実装では差異が生じ得ます。
- 提案: キャッシュ鍵は `params_signature()` の量子化値で作る一方、実行引数はランタイム解決済みの非量子化値を渡すように `G` 実装を分離する（鍵用と実行用の値を保持）。

2) パイプライン鍵「詳細仕様」の記述差
- 仕様（architecture.md）: 「詳細」節では pipeline_key を「関数 `__code__.co_code` の blake2b-64 と、パラメータの“決定的 repr” blake2b-64 を積む」と記載。
  - 該当: `architecture.md:227`
- 実装: `params_signature()` による「量子化＋ハッシュ可能化したタプル」を blake2b-64 に圧縮して使用。
  - 該当: `src/api/effects.py:116-122`
- 影響: アーキ文書内で「高レベルの説明（量子化署名ベース）」と「詳細（repr ベース）」が混在。実装は前者（量子化署名ベース）。
- 提案: architecture.md の「詳細」節を実装に合わせて「量子化署名（`common.param_utils.params_signature`）の blake2b-64」に統一。

## 不一致（軽微 / ドキュメンテーション）

3) 量子化ステップの既定値（docstring の記述）
- 仕様（architecture.md）: float の量子化既定は 1e-6（環境変数 `PXD_PIPELINE_QUANT_STEP` で上書き可）。
  - 該当: `architecture.md:248`
- 実装: コード上の既定も 1e-6（整合）。
  - 該当: `src/common/param_utils.py:45-53`
- 但し: `quantize_params()` の docstring に 1e-3 と記されており記述が古い。
  - 該当: `src/common/param_utils.py:78-92`
- 提案: docstring の既定値表記を 1e-6 に更新。

## 参考（整合確認ログ）
- `engine/*` から `api/*`/`effects/*`/`shapes/*` の import は検出されず（docstring 例外のみ）。
  - 例: `src/engine/core/geometry.py:63` の `from api import G, E` は docstring 内の使用例で実行コードではありません。
- 投影行列/ブレンド/MSAA/インデックス付与は下記参照。
  - 行列: `src/api/sketch.py:292`
  - ブレンド: `src/api/sketch.py:263-268`
  - MSAA: `src/engine/core/render_window.py:28-36`
  - IBO 生成: `src/engine/render/renderer.py:121-146`

## 次アクション（提案）
- Shapes 実行引数の量子化を外す（鍵のみ量子化）か、ドキュメントを現状実装に合わせるかの方針決定。
- architecture.md の pipeline_key 詳細説明を「量子化署名ベース」に統一。
- `param_utils.quantize_params` の docstring 既定値を 1e-6 に修正。

本レポートは読み取りと静的解析に基づくもので、動的な副作用や環境依存分岐は対象外です。必要であれば実行テストで補完してください。

