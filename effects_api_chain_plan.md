# Effects API（E 直後チェーン）改善計画

目的: `E.pipeline.affine().fill()` ではなく `E.affine().fill()` のように、`E.` 直後からエフェクトメソッドチェーンを開始できるようにする。
現状の `E.pipeline` API との互換性は維持しつつ、新スタイルを第一級の書き方として整理する。

## スコープ

- 対象:
  - `src/api/effects.py`（`_EffectsAPI`, `PipelineBuilder`, `Pipeline`）
  - スタブ生成: `tools/gen_g_stubs.py` → `src/api/__init__.pyi`
  - 関連ドキュメント・コメント: `architecture.md`, `docs/spec/pipeline.md`, `README.md`, `src/api/AGENTS.md` など
  - 既存テスト追加/更新: `tests/api/`, `tests/ui/parameters/`、必要に応じて `tests/smoke/`
- 非対象（今回やらないこと）:
  - エフェクト実装本体（`src/effects/*`）の設計変更
  - Parameter GUI の UI デザイン変更（カテゴリ名・色・レイアウトなど）
  - 署名生成や LazyGeometry の根本仕様変更

## チェックリスト（大項目）

- [x] A. API 仕様の決定と整理
- [x] B. 実装変更（`src/api/effects.py`）
- [ ] C. スタブ生成と型まわりの追従
- [ ] D. ドキュメント・コメントの更新
- [ ] E. テスト追加・既存テストの確認
- [ ] F. 既存スケッチ/デモへの影響確認・必要に応じたサンプル更新
- [ ] G. 開発者向けメモ・今後の拡張アイデア
- [ ] H. Parameter GUI 関連の注意点整理

以下で各項目を細分化する。

---

## A. API 仕様の決定と整理

目的: 実装に入る前に、`E` の公開 API を明確化する。

- [x] A-1. `E` のトップレベルでサポートするメソッドを決める
  - 候補:
    - `E.<effect>(...) -> PipelineBuilder`
    - `E.cache(maxsize=...) -> PipelineBuilder`
    - `E.label(uid=...) -> PipelineBuilder`
    - `E.pipeline` は「レガシー/低レベル」な開始点として残す
  - 採用: `E.<effect>(...)` と `E.pipeline`、`E.label(uid=...)` をトップレベルに用意し、`E.cache` は提供しない（ビルダチェーンでのみ使用）。
- [x] A-2. `E.<effect>` 呼び出しの振る舞いを定義
  - 新規ビルダを生成して最初のステップを追加し、その `PipelineBuilder` を返す
  - 例: `E.affine(...).fill(...)` は `E.pipeline.affine(...).fill(...)` と同じ steps を持つこと
- [x] A-3. `cache` / `label` などビルダユーティリティの入り口をどうするか決める
  - 案1: `E.cache(...).affine()` のように書けるようにする
  - 案2: ユーティリティは `E.pipeline.cache()` 経由に限定し、`E.<effect>` はエフェクト専用と割り切る
  - 採用: `label` は `E.label(uid)` と `PipelineBuilder.label(uid)` の両方から利用可能とし、`cache` は `PipelineBuilder.cache()` のみで提供。
- [x] A-4. `E` に対する誤用パターンの扱い方を決める
  - 例: 存在しないエフェクト名 `E.unknown_fx(...)` を呼んだときの例外形
- [x] A-5. 既存の `E.pipeline` の位置づけを決める
  - ドキュメント上で「後方互換のため残すが、新規コードでは `E.<effect>` を推奨」とするかどうか
- [x] A-6. Parameter GUI/署名/キャッシュに影響を与えないことを前提条件として明文化

**要: ユーザー確認ポイント 1**  
→ 上の A 系チェック項目と方針案で問題ないか、追記・削除したい観点があれば教えてください。

---

## B. 実装変更（`src/api/effects.py`）

目的: `_EffectsAPI` を拡張して `E.affine().fill()` 形式を実現する。

- [x] B-1. 現行 `_EffectsAPI` と `PipelineBuilder` の責務を再確認
  - `_EffectsAPI.pipeline` プロパティが単に `PipelineBuilder()` を返していることを再確認
  - `PipelineBuilder.__getattr__` のフロー（runtime → param resolve → `steps` 追加）を把握
- [x] B-2. `_EffectsAPI` に `__getattr__` を追加する実装方針を固める
  - `E.<name>` アクセス時に `PipelineBuilder()` を作成し、そのビルダの `<name>` メソッドをラップして返す形を想定
  - `E.pipeline` との衝突を避けるため、`name == "pipeline"` は特別扱いする
- [x] B-3. `__getattr__` 実装の詳細設計
  - 例外処理ポリシー: 存在しない effect 名に対する扱い（`effects.registry.get_effect` の失敗タイミング）
  - `dir(E)` 等の挙動に対する影響を最小限にとどめるかどうか
- [x] B-4. `cache` / `label` の扱いを決めて実装
  - 採用する仕様（A-3 で決めた内容）に従い、`E.label` をトップレベルに、`cache` は `PipelineBuilder` 専用とする。
  - 実装: `E.__getattr__` から常に `PipelineBuilder.__getattr__` を経由しつつ、`E.label(uid)` は `PipelineBuilder().label(uid)` を返す薄いファサードとして実装。
- [x] B-5. 実装後の自己レビュー
  - 簡潔で循環依存を起こさないこと
  - 例外メッセージ・エラーパスが不自然でないこと
  - `PipelineBuilder.__getattr__` を必ず経由するようにして、ParameterRuntime と署名生成のフローを変えないこと

---

## C. スタブ生成と型まわりの追従

目的: IDE 補完と mypy 等で `E.<effect>` 形式を正しく扱えるようにする。

- [x] C-1. `tools/gen_g_stubs.py` の `_render_pipeline_protocol` を読み直す
  - 現状: `_PipelineBuilder` に effect メソッド、`_Effects` に `pipeline()` のみ生成
- [x] C-2. `_Effects` Protocol にも effect メソッドを生やす方針を決める
  - 例: `_Effects` が `_PipelineBuilder` と同じメソッド群を持つようにする
- [x] C-3. 実装案
  - `_PipelineBuilder` 生成ループを共通化し、同じメソッド定義を `_Effects` にも出力
  - `E.<effect>` の戻り値型を `_PipelineBuilder` とする（チェーン可能にする）
- [x] C-4. スタブ生成スクリプトを修正し、`python -m tools.gen_g_stubs` で `src/api/__init__.pyi` を再生成
- [x] C-5. スタブ同期テストの確認
  - `tests/stubs/test_g_stub_sync.py`
  - `tests/stubs/test_pipeline_stub_sync.py`
- [ ] C-6. mypy 実行（必要に応じて）

---

## D. ドキュメント・コメントの更新

目的: 公開仕様として「`E.<effect>` 推奨」を反映しつつ、`E.pipeline` との関係を明示する。

- [x] D-1. `architecture.md` の Effects セクション更新
  - 現記述: 「`E.pipeline.<name>(...)` でチェーンし…」
  - 変更案: 「`E.<name>(...)` でチェーンし…（`E.pipeline.<name>` も後方互換として利用可能）」など
- [x] D-2. `docs/spec/pipeline.md` の基本 API 記述更新
  - 「ビルダー開始: `E.pipeline`」→ `E.<effect>` と `E.pipeline` 両方の例示
- [x] D-3. `README.md` のコード例更新
  - `E.pipeline.rotate(...).fill(...).build()` → `E.rotate(...).fill(...).build()` をメイン例に
- [x] D-4. `src/api/AGENTS.md` の説明文更新
  - 「`E.pipeline`（エフェクトパイプライン）」の文言を `E`/`E.<effect>` ベースに調整
- [x] D-5. コード内コメントと docstring のチェック
  - `src/engine/core/geometry.py` の使用例など、`E.pipeline` 前提の記述を更新
- [ ] D-6. `docs/reviews/*` や `docs/spec/*` 内の参照箇所を可能な範囲で追従

---

## E. テスト追加・既存テストの確認

目的: 新 API で既存の挙動（バイパス・キャッシュ・署名など）が保持されることを確認する。

- [x] E-1. 単体テスト（API レベル）の追加
  - 例: `tests/api/test_pipeline_spec_and_strict.py` に `E.rotate(...).build()` 経由のケースを追加
- [x] E-2. バイパス挙動のテスト
  - `tests/ui/parameters/test_effect_bypass.py` に `E.scale(bypass=True)` 形式のケースを追加
  - ParameterRuntime 有効時に `E.scale()` 経由でも `bypass` が GUI から効くことを確認
- [x] E-3. キャッシュ挙動のテスト
  - `tests/api/test_pipeline_cache.py` に `E.rotate(...).cache(...)` を使うケースを追加（仕様により）
- [x] E-4. 既存 `E.pipeline` テストがすべて通ることを確認
  - 新 API 追加により既存の仕様が変化していないか確認
- [ ] E-5. smoke テスト・簡易 pytest 実行
  - `pytest -q -m smoke` または対象ファイル限定で実行

---

## F. 既存スケッチ/デモへの影響確認・サンプル更新

目的: ユーザー向けコード例を新 API で統一し、実際の使用感を確認する。

- [ ] F-1. `sketch/25*.py` の使用状況確認
  - `E.pipeline` を使っている箇所の洗い出し（必要に応じて一部だけ新スタイルに更新）
- [ ] F-2. `demo/01.py` や `demo/define_your_effect.py` の書き方を `E.<effect>` ベースに更新
- [ ] F-3. 新旧書き方の共存方針を決める
  - 例: ドキュメントは新スタイル、テスト・一部内部コードは `E.pipeline` のままなど

---

## G. 開発者向けメモ・今後の拡張アイデア

目的: 実装中に気づいた点や、追加で検討したい事項をメモとして残す。

- [ ] G-1. 実装中に発見した注意点・落とし穴の追記
- [ ] G-2. `E` に他のユーティリティ（プリセットパイプラインなど）を生やす案の検討メモ
- [ ] G-3. 将来的に `E.pipeline` を非推奨化する場合の段階的な移行案メモ

---

## H. Parameter GUI 関連の注意点整理

目的: `E.<effect>` 形式にしても Parameter GUI の振る舞い（カテゴリ名、bypass、署名など）が変化しないようにする。

- [x] H-1. `pipeline_uid`/`pipeline_label` 生成タイミングの維持
  - `PipelineBuilder.__getattr__` 内でのみ `runtime.next_pipeline_uid()` を呼ぶ現在の仕様を維持し、`E.__getattr__` 側では UID を生成しない。
  - `E.<effect>` でも、最初のエフェクトステップ追加時にだけ UID が確保されることを確認する。
- [x] H-2. Parameter GUI のカテゴリ名が変わらないことの確認
  - `engine.ui.parameters.runtime.ParameterRuntime.before_effect_call` 内の `category` 決定ロジック（`pipeline_label or pipeline_uid or scope`）が、`E.<effect>` でも同じ値になることを確認する。
  - `E.label("Foo").affine(...)` のようなケースで、`label()` が最初のエフェクト追加前に呼ばれ、`pipeline_label` として GUI に反映されることをテストする。
- [x] H-3. `bypass` パラメータと Descriptor ID の整合性
  - `ParameterRuntime.before_effect_call` の末尾で登録される `bypass` の Descriptor ID（例: `effect@p0.scale#0.bypass`）が、`E.pipeline.scale()` と `E.scale()` の両方で同一規則になることを確認する。
  - `tests/ui/parameters/test_effect_bypass.py` に `E.scale()` 形式のケースを追加し、GUI override が効くことと Descriptor の ID が期待通りであることを検証する。
- [x] H-4. Parameter GUI からの `t` 自動注入の確認
  - `ParameterRuntime.before_effect_call` が effect シグネチャに基づき `t` を自動で追加する挙動が、`E.<effect>` 経由でも変わらないことを確認する。
- [ ] H-5. スナップショット/永続化との互換性
  - `engine.ui.parameters.snapshot` 周辺で利用している `pipeline_uid`/`pipeline_label` が、新 API 追加後も同じ意味で解釈されることを簡単に確認する（重大な変更が発生しないことを前提）。
  - 必要であれば Parameter GUI 向けの簡易スモークテストを追加し、`E.<effect>` 形式を用いたスケッチで GUI パラメータが期待通り表示・操作できることを確認する。

---

## 確認してほしいこと

- 上記チェックリストの粒度や範囲は、この改善の目的に対して過不足ないでしょうか。
- 特に A（仕様決定）や D（ドキュメント更新）で「ここは必ず明記してほしい」という点があれば教えてください。
- 実装優先度（たとえば「まずスケッチ/デモを新スタイルに揃したい」など）の希望があれば、この md に追記した上で着手順を調整します。
