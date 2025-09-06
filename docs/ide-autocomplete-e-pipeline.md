**E.pipeline 連鎖補完計画（fluent API の静的補完）**

目的
- `from api import E` 利用時に、`E.pipeline.rotate(...).fill(...).build()` のようなメソッド連鎖で、
  - `rotate`, `fill` などの登録エフェクト名が補完候補に出る
  - 各エフェクトの引数名・型ヒントが補完に表示される
  - 連鎖が続く（戻り値が PipelineBuilder と解釈される）

背景（現状と課題）
- ランタイムは `PipelineBuilder.__getattr__` による動的解決で柔軟性あり。
- しかし、Pylance/PyCharm の静的補完は動的属性を列挙できないため、エフェクト名や引数補完が出にくい。

採用方針（G と同様の .pyi 自動生成）
- `.pyi` に Protocol を定義し、`E.pipeline` の戻り値を「全エフェクトをメソッドとして持つビルダー」として静的宣言する。
- エフェクト登録は `effects.registry` に集約済みのため、レジストリを走査して `.pyi` を自動生成する。

設計概要
- `_PipelineBuilder` Protocol（自動生成）
  - 各エフェクトを `def <name>(self, *, <sig> , **_params: Any) -> _PipelineBuilder: ...` で宣言。
    - 引数は `effects.<fn>(g: Geometry, *, ...)` の `g` を除いたキーワード専用部を反映。
    - 既存実装との互換性のため、将来拡張用に `**_params: Any` を許容。
  - 付帯メソッドを手書きで定義: `cache(*, maxsize: int | None) -> _PipelineBuilder`, `strict(enabled: bool = True) -> _PipelineBuilder`, `build() -> Pipeline`, `__call__(g: Geometry) -> Geometry`。
- `_Effects` Protocol（手書き）
  - `@property def pipeline(self) -> _PipelineBuilder: ...` を宣言。
- `api/__init__.pyi` で `E: _Effects` とする（既存の `E as E` を置換）。

実装ステップ
1) スタブ生成ロジックの追加（新規/既存スクリプト拡張）
- `scripts/gen_g_stubs.py` を拡張 or `scripts/gen_pipeline_stubs.py` を新設。
- 実装要点:
  - `from effects.registry import list_effects, get_effect` で関数を取得。
  - `inspect.signature` / `typing.get_type_hints` で引数型を収集。先頭 `g` は除外。
  - 生成フォーマット例:
    ```python
    class _PipelineBuilder(Protocol):
        def rotate(self, *, pivot: Vec3 = ..., angles_rad: Vec3 = ..., **_params: Any) -> _PipelineBuilder: ...
        def fill(self, *, density: float = ..., pattern: str = ..., **_params: Any) -> _PipelineBuilder: ...
        def build(self) -> Pipeline: ...
        def strict(self, enabled: bool = ...) -> _PipelineBuilder: ...
        def cache(self, *, maxsize: int | None) -> _PipelineBuilder: ...
        def __call__(self, g: Geometry) -> Geometry: ...

    class _Effects(Protocol):
        @property
        def pipeline(self) -> _PipelineBuilder: ...

    E: _Effects
    ```

2) 生成先と統合
- `api/__init__.pyi` に `_PipelineBuilder` / `_Effects` を追記し、`E: _Effects` を出力。
- 既存の `G` の Protocol と共存させる（同一 `.pyi` 内に併記）。

3) テスト（同期検証）
- `tests/test_pipeline_stub_sync.py` を追加。
- `effects.registry.list_effects()` と `.pyi` のメソッド列挙が一致することを検証。
  - シグネチャは「関数名が存在すること」を主検証に留め、詳細型の厳密一致は緩める（`Any` 許容）。

4) 開発フロー組み込み
- pre-commit: 既存の `gen-g-stubs` にパイプライン生成を統合（または別フックを追加）。
- CI: 既存のワークフローで `.pyi` 差分チェックと同期テストを追加実行。

考慮点
- 可変キーワード（`**kwargs`）の扱い: 元エフェクトが `**kwargs` を許容する場合、`.pyi` メソッド末尾の `**_params: Any` に吸収させる。
- 依存のダミー注入: `numba` や重依存が無い環境でも生成が落ちないよう、`gen_g_stubs.py` と同様にダミーを注入。
- 名前正規化: `effects.registry` のキー正規化ポリシー（`BaseRegistry._normalize_key`）に合わせる。Python 識別子でない名前は除外。

既知の制約
- 実行時に外部プラグインで登録されたエフェクトは `.pyi` に反映されない（コア内蔵のみ対象）。
- 一部エフェクトの型ヒントが欠ける場合は `Any` となる。

ロールアウト
- 1) 生成ロジック実装 → 2) `.pyi` 再生成コミット → 3) 同期テスト/CI 追加 → 4) README 追記。

将来拡張
- パラメータ候補（choices）を doc コメントとして `.pyi` に埋め込み（`__param_meta__['choices']` など）。
- `to_spec/from_spec/validate_spec` の戻り値・引数型も `.pyi` で厳密化。

**進捗チェックリスト**

- [x] 計画ドキュメント作成（本ファイル）

- 生成スクリプト
- [x] 方針決定：既存の `scripts/gen_g_stubs.py` を拡張
- [x] エフェクト列挙：`effects.registry.list_effects()` から登録名を取得
- [x] シグネチャ抽出：`inspect.signature`/`get_type_hints`（`g` を除外・KW専用化・既定値は `...`）
- [x] 互換キーワード：`**_params: Any` を末尾に付与
- [x] 依存回避：`numba`/`fontTools` 未導入でも動くダミー依存注入を実装

- `.pyi` 出力
- [x] `_PipelineBuilder` Protocol を生成（各エフェクトをメソッド化、戻り値は `_PipelineBuilder`）
- [x] 付帯メソッドを宣言：`build`/`strict`/`cache`/`__call__`
- [x] `_Effects` Protocol を宣言し、`@property pipeline -> _PipelineBuilder`
- [x] `api/__init__.pyi` に `_PipelineBuilder` / `_Effects` を追記し、`E: _Effects` を宣言

- テスト
- [x] `tests/test_pipeline_stub_sync.py` を追加（`list_effects()` と `.pyi` のメソッド名が一致）
- [x] ローカルでテスト実行（pytest）

- 自動化
- [x] pre-commit にパイプライン同期テストを追加
- [x] GitHub Actions にパイプライン同期テストを追加

 - ドキュメント
 - [x] README の「IDE 補完」節に `E.pipeline` 連鎖補完を追記
 - [x] `docs/guides/typing.md` に `E.pipeline` の扱いを追記

 - ロールアウト
 - [x] 初回の `.pyi` 生成を実行（コミットは別途）
 - [x] エディタ反映トラブルシュート（VS Code/PyCharm）を README に転載

- 将来拡張（任意）
   - [x] `__param_meta__['choices']` を `.pyi` にコメントとして出力（各メソッド直前の `# choices:` 行）
   - [x] `min/max/type` を `.pyi` に `# meta: param (type=..., range=[min, max])` 形式でコメント化
   - [x] `to_spec/from_spec/validate_spec` の型を `.pyi` で厳密化（`PipelineSpec`/`PipelineSpecStep`/`JSONLike` 定義）
