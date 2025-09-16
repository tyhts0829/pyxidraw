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
- [x] ファイル改名（git mv 相当）
  - [x] `src/api/pipeline.py` → `src/api/effects.py`
  - [x] `src/api/shape_factory.py` → `src/api/shapes.py`
  - [x] `src/api/runner.py` → `src/api/sketch.py`
- [x] クラス改名
  - [x] `Step` → `PipelineStep`（`effects.py` 内）
  - [x] `Effects` → `EffectsAPI`（シングルトン `E = EffectsAPI()`）
  - [x] `ShapeFactory` → `ShapesAPI`（シングルトン `G = ShapesAPI()`）
  - [x] （任意）`ShapeFactoryMeta` → `ShapesAPIMeta`
- [x] `src/api/__init__.py` のインポート/公開を更新
  - [x] `from .effects import E, from_spec, to_spec, validate_spec`
  - [x] `from .shapes import G, ShapesAPI`
  - [x] `from .sketch import run_sketch as run` / `run_sketch`
  - [x] `__all__` に `ShapesAPI` を反映（`EffectsAPI` はクラスとしては非公開のまま）
- [x] 型スタブ `src/api/__init__.pyi` を更新
  - [x] `from api.effects import Pipeline as Pipeline`
  - [x] `from .shapes import ShapesAPI as ShapesAPI`
  - [x] `from .sketch import run_sketch as run_sketch, run_sketch as run`
  - [x] `__all__` の整合
- [x] 参照箇所の一括置換
  - [x] `from api.runner import` → `from api.sketch import`
  - [x] `from api.pipeline import` → `from api.effects import`
  - [x] `from api.shape_factory import` → `from api.shapes import`
  - [x] ドキュメンテーション/例コード内の文字列・docstring を更新
- [x] 近接 AGENTS.md の整合
  - [x] `src/api/AGENTS.md` の例外許可モジュールを `runner.py` → `sketch.py` に更新
- [x] ルート `architecture.md` の参照更新（存在箇所のみ）
- [x] テストファイル名の更新（旧名→新名）
  - [x] `tests/api/test_runner_init_only.py` → `tests/api/test_sketch_init_only.py`
  - [x] `tests/api/test_runner_more.py` → `tests/api/test_sketch_more.py`
  - [x] `tests/api/test_shape_factory.py` → `tests/api/test_shapes_api.py`
- [x] スタブ再生成（必要に応じて）
  - [x] `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`
- [x] 変更ファイル限定のチェックを緑化
  - [x] `ruff check --fix {changed_files}`
  - [x] `black {changed_files} && isort {changed_files}`
  - [x] `mypy {changed_files}`
  - [x] 影響テスト（例）: 個別ファイル指定で実行（全緑）
