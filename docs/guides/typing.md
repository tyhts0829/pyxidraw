**型スタブとIDE補完の方針（簡易ガイド）**

目的
- `from api import G` で `G.sphere(...)` などの補完と型支援を提供する。

概要
- `api/__init__.pyi` を自動生成し、`G` / `E.pipeline` を Protocol として宣言。
  - `G`: 各シェイプを `Shape.generate()` のシグネチャでメソッド列挙。
  - `E.pipeline`: `_PipelineBuilder` として全エフェクトをメソッド列挙。`build/strict/cache/__call__` も宣言。
- 実行時は従来通り `__getattr__` による動的ディスパッチ。`.pyi` は型チェッカー専用で挙動には影響しない。

開発フロー
- 形状/エフェクトの追加・削除・引数変更後に `python -m scripts.gen_g_stubs` を実行（G/E 同時に更新）。
- pre-commit/CI で自動検証（未同期で失敗）。

設計メモ
- 引数はキーワード専用（`*`）に統一し、`**_params: Any` を末尾許容して将来拡張に備える。
- Python 識別子でないシェイプ名は `.pyi` から除外（命名は `snake_case` を推奨）。
- 生成時に `numba`/`fontTools` 未導入でも落ちないようダミー依存を注入（スクリプト内、実行時に影響なし）。

エディタ反映のコツ
- VS Code: 反映されない場合は「Developer: Reload Window」。
- PyCharm: 「Invalidate Caches / Restart」。

配布
- パッケージ化する場合は `api/__init__.pyi` を含める（`MANIFEST.in` に `include api/__init__.pyi`）。

既知の限界
- 各シェイプの引数型が未注釈の場合は `Any` となる。
- 実行時登録の外部カスタムシェイプは補完対象外（コア内蔵のみ自動化）。
**型スタブと IDE 補完（ガイド）**

目的
- `from api import G` で `G.sphere(...)` などの補完・ホバー説明・型支援を提供する。

仕組み概要
- 自動生成する `api/__init__.pyi` に `G: _GShapes` を宣言し、登録シェイプをメソッドとして列挙。
- 各メソッドはブロック形式のスタブ（関数 docstring + 本体 `...`）。docstring は元シェイプ `generate()` の docstring から自動抽出され、IDE のホバーに表示される。
- 実行時は従来通り `__getattr__` による動的ディスパッチ。`.pyi` は型チェッカー専用で、ランタイム挙動は不変。

生成・検証フロー
- 形状を追加/削除/引数変更したら実行: `python -m scripts.gen_g_stubs`
- pre-commit で自動生成と同期テスト（`tests/test_g_stub_sync.py`）を実行。
- CI でも `.pyi` の同期を検証（diff 検出で失敗）。

docstring 埋め込みの仕様
- 対象: 各シェイプクラスの `generate()` の docstring。
- 抽出:
  - 要約: 先頭の非空 1 行を要約として使用。
  - 引数: `Args:` または `引数:` セクションを解析し、`name: 説明` 形式を抽出。複数行は前項目に連結。
  - 特殊引数: `**params` は `**_params` に正規化。
  - 短縮: 日本語句点（`。`/`．`）で区切り、長すぎる説明は適宜省略記号（…）。小数点（例: `0.5`）は保持。
- 出力: `_GShapes` の各メソッド直下に三重引用符の docstring を埋め込み、その後に本体 `...` を置く。

著者向け docstring 記述ガイド（推奨テンプレート）
```python
class Sphere(BaseShape):
    def generate(self, subdivisions: float = 0.5, sphere_type: float = 0.5, **_params: Any) -> Geometry:
        """Generate a sphere with radius 1.

        Args:
            subdivisions: Subdivision level (0.0-1.0, mapped to 0-5)
            sphere_type: Drawing style (0.0-1.0): Lat-Lon / Wireframe / ...
            **_params: Additional parameters (ignored)
        """
        ...
```
- ポイント:
  - 1 行目に要約（名詞始まりか動詞始まり、どちらでも可）。
  - `Args:` を使い、各行は必ず `name: 説明` の形式で開始。
  - 日本語の場合は一文目で完結させると見た目が良い（以降はスクリプトが…で省略）。

設計メモ
- 生成スタブの引数はキーワード専用（`*`）で統一。末尾に `**_params: Any` を許容して将来拡張に備える。
- Python 識別子でないシェイプ名は `.pyi` から除外（命名は `snake_case` を推奨）。
- 生成時に `numba`/`fontTools` 未導入でも落ちないようにダミー依存を注入（スクリプト内、実行時に影響なし）。

エディタでの反映
- VS Code (Pylance): 反映されない場合は「Developer: Reload Window」。
- PyCharm: 「File > Invalidate Caches / Restart」。

配布
- パッケージ配布時は `api/__init__.pyi` を必ず含める（例: `MANIFEST.in` に `include api/__init__.pyi`）。

既知の限界
- 未注釈の引数型は `Any` になる。
- 実行時にユーザーが追加したカスタムシェイプ（外部プラグイン）は補完対象外（コア内蔵のみ自動化）。
