# API 命名変更一覧（案・対称性重視）

範囲: `src/api`（engine 側は別スコープで後続対応）

## モジュール名の変更
- `src/api/pipeline.py` → `src/api/effects.py`
- `src/api/shape_factory.py` → `src/api/shapes.py`
- `src/api/runner.py` → `src/api/sketch.py`

## クラス名の変更
- `src/api/pipeline.py`（→ `src/api/effects.py` 内）
  - `Step` → `PipelineStep`
  - `Pipeline` → 変更なし（維持）
  - `PipelineBuilder` → 変更なし（維持）
  - `Effects` → `EffectsAPI`（`E = EffectsAPI()` の公開シングルトンは維持）

- `src/api/shape_factory.py`（→ `src/api/shapes.py` 内）
  - `ShapeFactory` → `ShapesAPI`（`G = ShapesAPI()` の公開シングルトンは維持）
  - `ShapeFactoryMeta` → `ShapesAPIMeta`（任意・対称性のため）

## 公開エクスポート/インポートの変更（`src/api/__init__.py`）
- `from .pipeline import E, from_spec, to_spec, validate_spec`
  → `from .effects import E, from_spec, to_spec, validate_spec`
- `from .shape_factory import G, ShapeFactory`
  → `from .shapes import G, ShapesAPI`
- `from .runner import run_sketch as run` / `run_sketch`
  → `from .sketch import run_sketch as run` / `run_sketch`
- `__all__` 更新:
  - `"ShapeFactory"` → `"ShapesAPI"`
  - 必要に応じ `"EffectsAPI"` を追加（クラス自体を公開する場合）

## 型スタブの変更（`src/api/__init__.pyi`）
- `from api.pipeline import Pipeline as Pipeline`
  → `from api.effects import Pipeline as Pipeline`
- `from .shape_factory import ShapeFactory as ShapeFactory`
  → `from .shapes import ShapesAPI as ShapesAPI`
- 既存の `G: _GShapes` / `E: _Effects` は維持（公開シングルトンの型は Protocol で表現）
- `from .runner import run_sketch` を `from .sketch import run_sketch` に更新
- 必要に応じて `ShapesAPI` を公開名として追加（`__all__` 整合）

## 互換エイリアス方針
- 互換エイリアスは作成しない（破壊的変更を許容する）。

## メモ
- デコレータ/レジストリは現状の対称性が良いため変更なし: `@effect` / `@shape`、`effects.registry` / `shapes.registry`。
- エンジン側の名称衝突は別スコープで整理（例: `engine/pipeline/*` → `engine/runtime/*` 等）。

---

## 追加方針（2025-09-15）
- `sketch_runner` ではなく短名 `sketch` とする。
- 互換シム/別名を設けず、全面リネームで統一する（破壊的変更可）。

---

## 実行チェックリスト（アクションリスト）
- [ ] ファイル改名（git mv）
  - [ ] `src/api/pipeline.py` → `src/api/effects.py`
  - [ ] `src/api/shape_factory.py` → `src/api/shapes.py`
  - [ ] `src/api/runner.py` → `src/api/sketch.py`
- [ ] クラス改名
  - [ ] `Step` → `PipelineStep`（`effects.py` 内）
  - [ ] `Effects` → `EffectsAPI`（シングルトン `E = EffectsAPI()`）
  - [ ] `ShapeFactory` → `ShapesAPI`（シングルトン `G = ShapesAPI()`）
  - [ ] （任意）`ShapeFactoryMeta` → `ShapesAPIMeta`
- [ ] `src/api/__init__.py` のインポート/公開を更新
  - [ ] `from .effects import E, from_spec, to_spec, validate_spec`
  - [ ] `from .shapes import G, ShapesAPI`
  - [ ] `from .sketch import run_sketch as run` / `run_sketch`
  - [ ] `__all__` に `ShapesAPI`（必要なら `EffectsAPI`）を反映
- [ ] 型スタブ `src/api/__init__.pyi` を更新
  - [ ] `from api.effects import Pipeline as Pipeline`
  - [ ] `from .shapes import ShapesAPI as ShapesAPI`
  - [ ] `from .sketch import run_sketch as run_sketch, run_sketch as run`
  - [ ] `__all__` の整合
- [ ] 参照箇所の一括置換
  - [ ] `from api.runner import` → `from api.sketch import`
  - [ ] `from api.pipeline import` → `from api.effects import`
  - [ ] `from api.shape_factory import` → `from api.shapes import`
  - [ ] ドキュメンテーション/例コード内の文字列・docstring を更新
- [ ] 近接 AGENTS.md の整合
  - [ ] `src/api/AGENTS.md` の例外許可モジュールを `runner.py` → `sketch.py` に更新
- [ ] ルート `AGENTS.md` / `architecture.md` の参照更新（存在箇所のみ）
- [ ] スタブ再生成
  - [ ] `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`
- [ ] 変更ファイル限定のチェックを緑化
  - [ ] `ruff check --fix {changed_files}`
  - [ ] `black {changed_files} && isort {changed_files}`
  - [ ] `mypy {changed_files}`
  - [ ] 影響テスト（例）: `pytest -q -k "api and (effects or shapes or sketch)"`

