# リファクタ計画: api.shapes — 量子化仕様準拠とドキュメント整合

目的: Shapes API の「量子化はキャッシュ鍵のみ、実行は非量子化」の仕様に厳密に合わせ、実装とドキュメントの不一致を解消する。ついでに軽微なドキュメント整合と未使用コードの整理を行う。

背景（現状の課題）
- 仕様: Shapes はキャッシュ鍵（署名）生成に量子化を用いるが、実行には非量子化の最終値を渡す（architecture.md:336）。
- 実装: `src/api/shapes.py` では量子化済みの `params_tuple` を `LazyGeometry.base_payload` に格納し、そのまま実行へ渡ってしまう経路がある。
  - 関連箇所: `src/api/shapes.py:112-118`（`_lazy_shape` が `params_tuple` を dict 化して payload に保存）、`src/api/shapes.py:160-169`（署名作成と `_lazy_shape` 呼び出し）。
- ドキュメント不一致: モジュール冒頭の LRU 説明が実装と乖離（実際は `engine.core.lazy_geometry` に形状結果 LRU がある）。`_build_shape_method` の戻り値 docstring も型不一致（実戻りは `LazyGeometry`）。
- 未使用コード: `_generate_shape_resolved` が未使用。

スコープ
- 変更ファイル: `src/api/shapes.py`
- 参照のみ: `src/engine/core/lazy_geometry.py`, `src/common/param_utils.py`, `src/shapes/registry.py`, `architecture.md`
- テスト: 既存 tests を流用（API 形状は非破壊）。必要に応じて最小追加テスト案を検討（任意）。

非目標（今回やらないこと）
- 公開 API 変更（関数名/引数形）は行わない → スタブ再生成は不要の見込み。
- キャッシュ方針や環境変数の仕様変更は行わない。

作業方針（要点）
1) 実行パラメータは非量子化値を Lazy payload に保持し、実行に渡す。
   - 署名（キャッシュ鍵）生成には引き続き `params_signature(impl, params)` を使用（量子化）。
   - 署名作成は `impl = getattr(fn, "__shape_impl__", fn)` を用いる（effects と同一方針）。
2) `_lazy_shape` は「非量子化 dict」を受け取り payload に保存する実装へ整理。
3) `_build_shape_method` は runtime 有/無の両ケースで「payload: 非量子化 dict」「spec 記録: 量子化署名」を両立させる。
4) ドキュメント整合（冒頭説明、戻り値表記、例の引数型修正）。
5) 未使用メソッド（`_generate_shape_resolved`）を削除。

実施手順（チェックリスト）
- [x] 署名生成対象を impl に統一（`src/api/shapes.py:160-169`）
- [x] `_lazy_shape` を「`params_dict: dict[str, Any]` をそのまま payload に保存」へ明確化（`src/api/shapes.py:112-118`）
- [x] `_build_shape_method` を更新
  - [x] runtime 有り: `resolved = runtime.before_shape_call(...)`（非量子化）→ payload へ。署名は `params_signature(impl, resolved)`。
  - [x] runtime 無し: `params`（非量子化）→ payload へ。署名は `params_signature(impl, params)`。
  - [x] `_record_spec(name, params_tuple)` は維持（統計用途）。
- [x] 未使用メソッド `_generate_shape_resolved` の削除
- [x] docstring 整合
  - [x] モジュール冒頭の LRU 記述を `engine.core.lazy_geometry` の shape 結果 LRU に合わせて簡潔に修正。
  - [x] `_build_shape_method` の戻り値を `Callable[..., LazyGeometry]` に修正。
  - [x] 例の `subdivisions=0.5` を `subdivisions=1` へ（整数例に統一）。
- [x] 影響確認（最小範囲）
  - [x] `ruff check --fix src/api/shapes.py`
  - [x] `mypy src/api/shapes.py`
  - [x] `pytest -q tests/api/test_shapes_api.py`

検証観点
- 量子化は LazyGeometry.realize() の shape 結果 LRU 鍵でも再計算されるため、今回の変更で鍵不一致が起きないこと（実引数と署名生成の入力が一致）。
- 既存テストが全て緑のこと（特に `tests/api/test_shapes_api.py::test_shape_factory_lru_returns_same_instance`）。

追加の任意テスト案（必要なら）
- 「量子化ステップ未指定時は 1e-6 デフォルト」前提で、`subdivisions` のような int パラメータは量子化の影響を受けず、そのまま実行に渡ることを確認する軽量テスト（JSON 互換の観測可能な副作用が乏しいため、既存テスト十分の可能性あり）。

リスクとロールバック
- リスク: 実行時に渡す値が変化する可能性（量子化→非量子化）。ただし仕様整合のため期待された振る舞い。
- ロールバック: `_lazy_shape` を元に戻し、`_build_shape_method` で量子化済み tuple を payload に戻す。

完了条件
- 上記チェックリスト完了。
- ruff/mypy/pytest（対象限定）が緑。
- ドキュメント（冒頭説明/戻り値/例）が現実の実装と一致。

備考
- 公開 API に変更はなく、スタブ再生成は不要の見込み。
