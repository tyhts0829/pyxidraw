# パラメータ二重正規化 問題整理と対応チェックリスト

## 指摘と問題の詳細
- 指摘: `src/engine/ui/parameters/value_resolver.py:155` で CLI/パイプライン経路の値が二重に正規化され、0.0〜1.0 入力が実レンジへ変換されずに残る。
- 症状: 例として `displace.amplitude_mm=0.5` を CLI から渡すと、本来 25.0 mm（0.0〜50.0 のレンジ）が期待されるが、実行時に 0.5 のまま effect 関数へ渡される。
- 影響: `resolve_without_runtime()` の docstring「0..1 入力を実レンジへ変換」と矛盾し、`architecture.md` が定義する「公開パラメータは正規化入力を実レンジへ変換する」設計と不一致。

## 対応アクション チェックリスト
- [x] `_resolve_scalar` の provided 経路から二度目の `normalize_scalar` 呼び出しを排除し、正規化済み入力が実レンジへ適切に変換されるよう修正する。
- [x] 既定値経路と override 経路での値変換が期待通りか確認し、必要であれば追加テストを設計する。
- [x] 二重正規化が解消されたことを `PYTHONPATH=src python - <<'PY' ...` のワンショット検証で確認する。
- [x] `tests/ui/parameters/test_value_resolver.py` など該当テストを追加/更新して 0.0〜1.0 入力が実レンジへ変換されることを自動検証する。
- [x] `ruff check --fix src/engine/ui/parameters/value_resolver.py` を実行し、必要に応じて `black`/`isort` で整形する。
- [x] `mypy src/engine/ui/parameters/value_resolver.py` を実行し、型チェックを通過させる。
- [x] `pytest -q tests/ui/parameters/test_value_resolver.py` を実行し、既存・追加テストの成功を確認する。
- [x] 変更内容に応じて `architecture.md` や関連ドキュメントに差分が生じた場合は同期する。
