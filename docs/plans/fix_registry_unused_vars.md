# レジストリ層の未使用変数(F841)修正チェックリスト

目的: ruff F841 (未使用変数) を解消し、変更範囲を最小に保つ。

対象:
- `src/effects/registry.py`: `eff_name = resolved_name` の未使用
- `src/shapes/registry.py`: `shp_name = resolved_name or orig_fn.__name__` の未使用

作業項目:
- [ ] `src/effects/registry.py` の未使用代入を削除（副作用なし）
- [ ] `src/shapes/registry.py` の未使用代入を削除（副作用なし）
- [ ] 変更ファイルに限定して ruff/black/isort/mypy を実行
  - `ruff check --fix src/effects/registry.py src/shapes/registry.py`
  - `black src/effects/registry.py src/shapes/registry.py && isort src/effects/registry.py src/shapes/registry.py`
  - `mypy src/effects/registry.py src/shapes/registry.py`
- [ ] 必要に応じて微調整（型/署名/メタ属性は現状維持の想定）

備考:
- 代入行の削除は既存の公開 API/振る舞いに影響しない想定。
- 将来、登録名をラッパにメタとして残す需要があれば `__registered_name__` などの属性として付与を検討（今回は未実施）。

