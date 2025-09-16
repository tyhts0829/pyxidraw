# コミット時の docstring 自動生成（スタブ生成）

目的: コミット前の pre-commit フックで `src/api/__init__.pyi`（公開 API スタブ）を再生成し、
`G`（形状ファクトリ）や `E.pipeline`（エフェクトビルダ）の各メソッドに要約と引数説明の
docstring を自動付与。IDE 補完/ツールチップと型検査の体験を高める。

- 生成スクリプト: `tools/gen_g_stubs.py`
- 出力先: `src/api/__init__.pyi`（自動生成ファイル、手編集しない）
- 起動タイミング: `.pre-commit-config.yaml` の `gen-g-stubs` フック（commit 前）
- 同期検証: `tests/test_g_stub_sync.py`, `tests/test_pipeline_stub_sync.py`

## 生成されるもの（What）

- `class _GShapes(Protocol)`
  - 登録済みシェイプ名ごとのメソッドを生成。
  - 署名はキーワード専用かつ寛容: `*, ..., **_params: Any`。
  - 各メソッド直下に docstring（要約/引数）を付与。
- `class _PipelineBuilder(Protocol)` / `class _Effects(Protocol)`
  - すべてのエフェクト関数をビルダメソッドとして公開。
  - 各メソッドに docstring（要約/引数）を付与。`build/strict/cache/__call__` も含む。
- JSON/Spec 型定義、`G: _GShapes`, `E: _Effects`、`Pipeline` 系の再エクスポート。

## 実装の流れ（How）

`tools/gen_g_stubs.py` の主役関数/責務:

- `generate_stubs_str()`
  - 依存の薄い環境でも動作するよう `tools.dummy_deps.install()` を先に実行。
- `effects`/`shapes` を import（レジストリ副作用を確実化）。
- `shapes.registry.list_shapes()` で全シェイプ名を取得し、Python 識別子のみ採用。
  - `_render_pyi(valid_names)` で最終文字列を構築して返す。
- `_render_pyi(shape_names)`
  - ヘッダ/共通 import/型別名を出力。
- 形状: 関数シグネチャから `G.<name>(...)` を生成（関数ベース統一後）。
  - エフェクト: `effects.registry.list_effects()` を列挙し、
    `_render_pipeline_protocol(effect_names)` でビルダ/Effects の Protocol 本体を生成。
  - 末尾で `G/E` や `run`、Spec ヘルパー関数を再エクスポート。
- 形状関数の解析
  - `inspect.signature` と `get_type_hints` から関数シグネチャを復元。
  - 既定値はスタブ上では `= ...` に正規化し API 表面を安定化。
  - 位置可変は無視、キーワード専用＋`**_params: Any` を常に付与（将来拡張/偽陰性回避）。
  - `_extract_param_docs(gen_obj)` で docstring を解析し、要約/引数説明をメソッド直下の
    三重引用符 docstring として埋め込む。
- `_extract_param_docs(gen_obj)`
  - 生成器の docstring（英語 `Args:`/日本語 `引数:` どちらも可）を寛容にパース。
  - 要約: 最初の非空行（セクション見出しは除外）。
  - 引数: `name: 説明` 形式を行ごとに抽出。続行行は連結し、120 文字で要約。
  - 特例: `**params` の説明はスタブの仮引数名 `**_params` に合わせて移し替え。
- `_render_pipeline_protocol(effect_names)`
  - 各エフェクト関数のシグネチャ/型ヒントからビルダメソッドを生成。
  - docstring は関数の docstring（要約/引数）を優先し、なければ `__param_meta__`
   （`{"type", "min", "max", "choices"}` を想定）から自動整形して記述。
  - `_annotation_for_effect_param()` で型注釈を簡素に描画（`Union` は `|`、`tuple[float, float, float]`
    は必要に応じて `common.types.Vec3` に写像し追加 import）。
- `main()`
  - `generate_stubs_str()` の結果を `src/api/__init__.pyi` に書き出し（自動上書き）。

## いつ動くか（pre-commit 連携）

`.pre-commit-config.yaml` 抜粋:

- `gen-g-stubs`: `PYTHONPATH=src python -m tools.gen_g_stubs`
- `test-g-stub-sync`: `pytest -q tests/test_g_stub_sync.py`
- `test-pipeline-stub-sync`: `pytest -q tests/test_pipeline_stub_sync.py`

コミット前にスタブを再生成し、直後に同期テストでディスク上の `api/__init__.pyi` と
生成文字列の完全一致を検証。一致しない場合はコミットを拒否する。

## 設計上の意図（Why / Trade-off）

- 依存の軽さ: 重い依存（numba/fontTools/shapely）未導入でも introspection が通るよう
  ダミー注入で堅牢化。CI/開発マシンの差異に影響されにくい。
- 型安全と柔軟性の両立: 既知パラメータは型注釈を描画、未知は常に `**_params: Any` で許容。
- 署名の安定化: 既定値評価の違いを避けるため、スタブでは `= ...` に正規化。
- ドキュメント品質: 要約/引数説明をスタブへ転記することで IDE のツールチップを強化。

## 書き方のガイド（作者が行うこと）

- 形状（関数）の docstring 例:

```python
from engine.core.geometry import Geometry

def sphere(*, radius: float = 1.0, segments: int = 64) -> Geometry:
    """球を生成。

    引数:
        radius: 半径。
        segments: 分割数。大きいほど滑らか。
    """
    ...
```

- エフェクト関数の docstring / メタ例:

```python
@effect
def rotate(g: Geometry, *, angle: float = 0.0, axis: tuple[float, float, float] = (0, 0, 1)) -> Geometry:
    """ジオメトリを回転。

    引数:
        angle: 角度 [deg]。
        axis: 回転軸ベクトル。
    """
    ...

# オプション: 検証情報があればメタで補強（docstring 不在時の説明生成にも利用）
rotate.__param_meta__ = {
    "angle": {"type": float, "min": -360, "max": 360},
    "axis": {"type": "Vec3"},
}
```

- 注意:
  - 見出しは「引数:」または「Args:」を使用。`name: 説明` で 1 行簡潔に書く。
  - `**params` を説明したい場合はそのまま書けば `**_params` に自動で移し替え。
  - 返り値/例外などは生成対象外（必要ならコード側 docstring に記述）。

## フォールバックと制限

- 形状/エフェクトの docstring が無い、または解析に失敗した場合:
  - メソッドは生成されるが、docstring は省略される（IDE の説明は簡素になる）。
- 形状名が Python 識別子でない場合（例: `foo-bar`）はスキップし、スクリプトが警告を出力。
- 型注釈の解決に失敗した場合は `Any` へフォールバック。

## 手動実行/トラブルシュート

- 手動実行:
  - `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`
- すべてのフックを通す:
  - `pre-commit run -a -v`
- 失敗時のチェックリストは `docs/pre-commit-troubleshooting.md` を参照。

## 参考（関連ファイル）

- スクリプト: `tools/gen_g_stubs.py`
  - 抽出: `_extract_param_docs()`
  - 形状: `_render_method_from_generate()`
  - エフェクト: `_render_pipeline_protocol()` / `_annotation_for_effect_param()`
- 設定: `.pre-commit-config.yaml`
- 検証: `tests/test_g_stub_sync.py`, `tests/test_pipeline_stub_sync.py`
