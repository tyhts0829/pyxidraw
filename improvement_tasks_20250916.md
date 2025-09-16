# 改善アクションチェックリスト

- [ ] `src/api/effects.py` に `import effects  # noqa: F401` を追加し、エフェクト登録の副作用を必ず発火させる。

  - 現状: `EffectsAPI` 側では `effects` パッケージを一度も import しておらず、`effects/__init__.py` が保有する登録副作用が走っていない。
  - 問題: `PipelineBuilder.build()` で `get_effect()` が即時呼び出されるため、レジストリ未初期化のまま `KeyError` が発生しユーザーコードが動作しない。
  - 影響: `from api import E` を使った最初のパイプライン構築が失敗し、API 一貫性（`G` 側は副作用 import 済み）も崩れている。
  - 確認結果: `src/api/effects.py:78` の `from effects.registry import get_effect` でパッケージ全体が読み込まれ、`effects/__init__.py` の副作用 import が発火しているためレジストリは初期化済み。`PYTHONPATH=src python - <<'PY' ...` による手動検証でも `E.pipeline.rotate().build()` は `KeyError` なく成功し、`sys.modules` に `effects.*` が展開されていることを確認。

- [x] `src` 配下に混入している `__pycache__` などのビルド生成物を削除し、`.gitignore` で再発を防ぐ。併せて実体のない `src/scripts/` の扱いを整理する。

  - 対応: `.gitignore` を整理して `__pycache__/`・`*.py[cod]`・`*$py.class` など Python 生成物の除外を明確化。`find`/Python スクリプトで 30 箇所の `__pycache__` を一括削除し、空になった `src/scripts/` ディレクトリを参照調査後に削除済み。
  - 備考: `git ls-files` では該当生成物はいずれも追跡されておらず、今回の作業は物理クリーンアップと再発防止設定の強化に留まる。

- [x] `Geometry` コンストラクタで `coords`/`offsets` の dtype・形状を検証し、不正入力を例外または正規化で吸収できるようにする。

  - 現状: `Geometry` は dataclass のフィールドに直接 ndarray を渡すだけで、dtype を float32/int32 に制約していない。
  - 問題: 利用者が手動で `Geometry(np.array(..., float64), np.array(..., int64))` を生成すると、内部で想定する 32bit 型とズレて後続処理（digest 計算や `concat` など）で余分なコピーやバグが発生するリスクがある。
  - 影響: 公開 API の契約が暗黙になり、静的解析やテストで見逃しやすい不正データがシステム内に入り込む可能性が高まる。

- [ ] `ShapesAPI` の動的ディスパッチ実装を簡素化し、`EffectsAPI` と対称性のある構造へリファクタリングする方針を検討する。
  - 現状: メタクラスで `setattr`・`staticmethod` を動的生成し、解除時に `delattr` する複雑なロジックを採用している。
  - 問題: 理解コストが高く、`EffectsAPI` の単純なプロパティベース設計と非対称で保守性が低い。`delattr` の例外処理なども潜在的なバグポイント。
  - 影響: チーム内での設計整合性が崩れ、将来の拡張やバグ調査時に実装意図を追うコストが増加する。

## メモ

- テスト追加方針や `.gitignore` の具体的な更新内容は要相談。
