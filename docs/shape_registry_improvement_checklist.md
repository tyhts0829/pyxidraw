［Legacy/Archived as of v6.0］
（注意）本ドキュメントは旧 API/旧ファイル名を前提にしたレビュー用メモです。
現行の名称対応:
- `api/shape_factory.py` → `api/shapes.py`
- `ShapeFactory` → `ShapesAPI`
- 実レジストリ参照は `shapes/registry.py`

本チェックリストは BaseShape 前提の旧 API 用です。v6.0 以降は shape は関数ベースに統一されました。
現行の設計/実装に合わせたレビューは `docs/proposals/shape_function_only_migration.md` を参照してください。

---

src/api/shape_registry.py コードレビューして

> 対象ファイルの内容を確認してから、観点別にレビューします。

• Explored
└ Search shape_registry.py
List ls -la
List src
List api
Read shape_registry.py, registry.py
Search 公開 API|shape_registry|shape factory|shape_registry.py|薄く委譲|再エクスポート in architecture.md
Read architecture.md, **init**.py
Search shape_registry
Read shapes.py, base.py, test_shape_registry_extra.py, gen_g_stubs.py, **init**.py

> 総評

- 役割は「shapes.registry への薄い委譲」を保っており、API 境界の設計意図に沿っています。命名・公開面（all）も最小限で良好です。
- ただし「公開 API に NumPy スタイル docstring + 型ヒント」のリポ規約に未準拠の点があり、mypy/Pylance での型推論が弱くなっています。特に get_shape_generator の戻り値型未指定により、上位の ShapeFactory で Any 化しているため、型安全性・補完品質を落としています。

良い点

- 責務が明確（KeyError→ValueError のマッピングのみ付加し、それ以外は委譲）。
- **all** を明示し、公開面を限定。
- レジストリの副作用初期化（shapes パッケージ import 経由）を満たす import 形態。
- テスト想定（未登録時に ValueError）に整合。

気になった点（改善余地）

- 型ヒント不足
  - get_shape_generator(name: str) に戻り値型がないため、呼び出し側で Any 扱いになりやすい。
  - unregister_shape(name: str) の戻り値型が未指定（実質 None）。
- Docstring 形式
  - 規約の「NumPy スタイル docstring」に未準拠（Parameters/Returns/Raises の節がない）。
  - list_registered_shapes はソート順（昇順）を明記すると、仕様が安定します。
- 命名の厳密さ
  - get_shape_generator は実際には「生成“器”関数」ではなく「シェイプ“クラス”」を返しています。命名と実体が軽くズレています（互換維持のため現状維持でも構いませんが、別名の併設を検討可）。
- import 形態の一貫性
  - unregister_shape だけモジュール経由（\_shapes_registry.unregister）で、他は関数を直接 import。どちらでも良いですが、統一すると可読性が上がります。

具体的な改善提案（互換を壊さず最小）

- 型ヒントの追加と NumPy スタイル docstring への更新
  - get_shape_generator の戻り値を type[BaseShape] に。
  - unregister_shape の戻り値を None に。
  - 各公開関数の docstring を NumPy スタイル化し、Raises/Returns を明示。

例（差分イメージ・導入後も既存テストは保持）:

- 変更ファイル: src/api/shape_registry.py
  - 先頭に from shapes.base import BaseShape を追加。
  - シグネチャ更新:
- docstring（NumPy スタイル）例:
  - get_shape_generator
    - 要約: 登録済みシェイプ“クラス”を取得。
    - Parameters: name: str
    - Returns: type[BaseShape]
    - Raises: ValueError
  - list_registered_shapes
    - Returns: list[str]（昇順ソート）
- 命名の補助（任意・非破壊）
  - 後方互換を維持しつつ、別名を追加:
  - `get_shape_class = get_shape_generator`
- 追加した場合は **all** にも併記（ドキュメント整備も忘れず）。

提案後の関数シグネチャ（抜粋・参考）:

- def get_shape_generator(name: str) -> type[BaseShape]: ...
- def list_registered_shapes() -> list[str]: ...
- def unregister_shape(name: str) -> None: ...
- register_shape = shape
- （任意）get_shape_class = get_shape_generator

影響/互換性

- 戻り値型注釈と docstring の追加はランタイム互換を壊しません。
- ShapeFactory 側で shape_cls: type[BaseShape] と解釈され、generate() 呼び出しの型安全性が向上します。
- もし get_shape_class エイリアスを追加しても、既存の get_shape_generator 呼び出しは維持されます。

# api/shape_registry 改善チェックリスト（提案）

目的: 型安全性とドキュメント整合（NumPy スタイル）を満たしつつ、挙動を一切変えずに可読性と一貫性を向上させる。

## 範囲（Scope）

- 対象: `src/api/shape_registry.py`
- 非対象: `src/shapes/registry.py` の挙動・公開 API、テスト仕様の挙動。

## 提案変更（最小・互換維持）

- 型注釈を強化
  - `get_shape_generator(name: str) -> type[BaseShape]`（現状は戻り値未注釈）
  - `unregister_shape(name: str) -> None`
- NumPy スタイル docstring（日本語）へ更新
  - Parameters / Returns / Raises を明示
  - `list_registered_shapes` の戻り順（昇順ソート）を仕様として明記
- import 形態の一貫化（どちらかに統一）
  - 案 A: すべて関数を直接 import して使用
  - 案 B: すべてモジュール別名（`_shapes_registry`）経由で呼び出し
- （任意）命名補助の別名を追加
  - `get_shape_class = get_shape_generator` を追加（実体はクラスを返すため）
  - 後方互換のため既存名は維持

## 非目標（Non-goals）

- 例外ポリシーの変更（`KeyError -> ValueError` マッピングは維持）
- 登録・取得ロジックの変更なし
- 公開 API 面の追加は（任意）別名のみ。デフォルトでは追加しない方針

## 作業チェックリスト（Do）

- [ ] 1. `BaseShape` の型参照を追加（`from shapes.base import BaseShape`）
- [ ] 2. `get_shape_generator` の戻り値注釈を `type[BaseShape]` に更新
- [ ] 3. `unregister_shape` の戻り値注釈を `None` に更新
- [ ] 4. 各関数の docstring を NumPy スタイル（日本語）に更新
- [ ] 5. import 形態の統一（下の質問の回答に従う）
- [ ] 6. （任意）`get_shape_class` エイリアス追加と `__all__` 更新
- [ ] 7. 変更ファイルに限定して `ruff/black/isort/mypy` 実行
- [ ] 8. 対象テスト（`tests/api/test_shape_registry_extra.py`）を実行
- [ ] 9. 影響範囲の最終確認（`scripts/gen_g_stubs.py` 参照部の型だけ確認）

## 受け入れ基準（DoD）

- 変更ファイルに対する以下が成功
  - `ruff check --fix src/api/shape_registry.py`
  - `black src/api/shape_registry.py && isort src/api/shape_registry.py`
  - `mypy src/api/shape_registry.py`
  - `pytest -q tests/api/test_shape_registry_extra.py`
- 挙動変更なし（既存テストがそのまま緑）
- 必要に応じて `__all__` が最新化

## 検証コマンド（編集ファイル優先・高速ループ）

```bash
ruff check --fix src/api/shape_registry.py
black src/api/shape_registry.py && isort src/api/shape_registry.py
mypy src/api/shape_registry.py
pytest -q tests/api/test_shape_registry_extra.py
```

## 確認事項（要回答）

1. import 形態の統一はどちらにしますか？
   - [ ] 案 A: 「関数を直接 import」（`from shapes.registry import unregister` 等）
   - [ ] 案 B: 「モジュール別名経由」（`_shapes_registry.unregister(...)` に揃える）
   - 推奨: 案 A（薄い委譲として読みやすく、他関数と統一できる）
2. 別名 `get_shape_class` を追加してもよいですか？（デフォルトは追加しない）
   - [ ] 追加する（`__all__` にも併記）
   - [ ] 追加しない（現状維持）

## 影響/互換性

- ランタイム挙動は不変。型注釈と docstring の追加のみ
- `ShapeFactory` など上位はより強い型恩恵（`type[BaseShape]`）を受ける
- `scripts/gen_g_stubs.py` の挙動は不変（呼び出し名/戻り値の実体は同じ）

## 備考

- `architecture.md` の公開面は `G/E/run/Geometry/...` に限定とあるため、本変更で更新不要と判断（必要なら追記可）

---

承認いただければ、チェックリストに沿って最小差分で実装・検証を進めます。
