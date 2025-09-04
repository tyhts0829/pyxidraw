# スマート・シンプル・可読性 向上チェックリスト（2025-09-04）

目的: リポジトリ全体をフラットに俯瞰し、「実装をよりスマートに・シンプルに・可読性を高く」するための具体的な改善項目を、優先度と影響面とともにチェックリスト化する。

補足: 既存のレビュー文書（例: `docs/refactor_review_2025-09-04.md`, `docs/code_review_2025-09-04.md`）と重複しないよう、横断的な“統一・簡素化・重複解消”に主眼を置いている。

---

## 即効（Quick Wins: 小さく安全、30分〜）
- [x] `engine/core/geometry.py`: 型ヒントの不足を補完（`from typing import Tuple` を追加）。
- [x] `engine/core/geometry.py`: `Geometry` に `is_empty` プロパティ（`return self.coords.size == 0`）を追加し、各所の一部を置換（可読性）。
- [x] `api/pipeline.Pipeline`: `__repr__`/`__str__` でステップ概略を返す（ログ/デバッグ容易化）。
- [x] `effects/translation.py`: 実装を `g.translate(...)` 委譲に統一（余剰の Numba 実装を撤去）。
- [x] `effects/rotation.py`: 未使用インポートの整理。
- [x] `engine/io/helpers.DualKeyDict`: `TOGGLE_CC_NUMBERS` をモジュール定数に昇格し、`__getitem__` の副作用（TODO）を撤去。

- [x] `Geometry` のダイジェスト計算を集中管理
  - 実施: モジュール内ヘルパ（`_set_digest_if_enabled`）を追加し、`translate/scale/rotate/concat/from_lines` の重複を削減。
  - [ ] `digest` 側の挙動を「無効化時は None を返さず、使用側がフォールバックする」方針に統一（現状は `RuntimeError` → `Pipeline` 側で広域 `except`）。専用 `try_digest()` の追加も可。（影響: 中）
- [ ] 変換 API の一本化
  - [x] `effects/translation.py` は `Geometry.translate` に委譲（Numba 実装を撤去）。
  - [x] `engine/core/transform_utils.py` は `transform_combined()` を主用途に、`translate/scale/rotate_*` を Geometry メソッドへの薄いラッパに変更（重複削減）。

## キャッシュ戦略のシンプル化（どこでキャッシュするかを一本化）
- [x] シェイプ生成のキャッシュ重複を解消
  - 実施: `shapes/base.py` の `BaseShape` は既定でキャッシュ無効化（`enable_cache=False`）。
    形状生成のキャッシュは `api/shape_factory.ShapeFactory._cached_shape` のみに一本化。
  - 方針A（採用）: Factory 側に一本化。必要なら各 Shape が `enable_cache=True` で個別最適を選択可能。
  - 方針B（見送り）: Factory から `__call__` を通す形は採用しない。
- [ ] ADR に「キャッシュはどの層の責務か」を追記して迷いを排除。（影響: 中）
  - [x] 実施: ADR 0011 を追加し、ShapeFactory/Shape/Pipeline の責務を明記。
- [ ] パイプライン単層キャッシュの既定値を環境変数で制御
  - 例: `PXD_PIPELINE_CACHE_MAXSIZE`（未設定: 無制限）を `PipelineBuilder.cache(maxsize=...)` のデフォルトに反映。（影響: 低）
  - [x] 実施: `PXD_PIPELINE_CACHE_MAXSIZE` を読み取り、0=無効/正整数=上限/None=無制限に対応。

## レジストリ/命名の一貫性
- [x] `effects/registry.py` と `shapes/registry.py` のキー正規化ルールを明文化
  - 現状: effects はハイフン→アンダースコア + lower、shapes は Camel→snake + lower など。
  - [x] ADR に「キー正規化ルール（ハイフン、CamelCase、大文字、小文字）」を一本化して記述。（影響: 低）
- [ ] エイリアス方針（非対応）を README/cheatsheet に明文化。
  - [x] `docs/effects_cheatsheet.md` にエイリアス非対応を明記。

## API/パラメータの読みやすさ（仕様を“見て分かる”に）
- [ ] `validate_spec()` とエフェクト関数の `__param_meta__` 利用範囲の整理
  - [x] `vec3` 型を `validate_spec` に追加（scalar/1-tuple/3-tuple を許容）
  - [x] 主要エフェクト（translate/rotate/scale）に `__param_meta__` を付与。
  - [x] 数値域・choices を含む表を `docs/effects_cheatsheet.md` に追加。（影響: 低）
- [ ] 角度・スケール・周波数など単位系の一貫化
  - [ ] README の方針に合わせ、Docstring に「単位（rad/mm/cycles per unit）」を明示（抜けている関数を補完）。（影響: 低）

## テスト/品質ツール（DX を上げながら保守コストを下げる）
- [ ] `effects/translation` と `Geometry.translate` の一致性テストを追加（同一入力→同一出力、digest 有効/無効の両系）。
- [ ] `Pipeline` のキャッシュ挙動（ヒット/ミス）をより網羅（既存テストにケース追加）。
- [ ] `PXD_DISABLE_GEOMETRY_DIGEST`=1 環境下でもキャッシュが安全に機能する回帰テストを追加。
- [x] `pyproject.toml` を追加し、`ruff`/`black`/`isort` の最小設定を導入。
- [ ] 未使用インポート/変数の検知ルール（ruff F401/F841）を有効化し、段階的に解消。

## ファイル/箇所別メモ（抜粋）
- `engine/core/geometry.py`
  - [ ] digest 制御の集中管理（env チェックの重複削減）。
  - [ ] `is_empty` 追加、`Tuple` import 追加。
- `effects/translation.py`
  - [ ] `g.translate` 委譲（Numba 実装は内部 util に退避 or 削除）。
- `engine/core/transform_utils.py`
  - [ ] `transform_combined()` を主とし、個別関数は薄いラッパ or 非推奨化。
- `api/shape_factory.py` / `shapes/base.py`
  - [ ] キャッシュ責務の一本化（Factory 側推奨）。
- `engine/io/helpers.py`
  - [ ] `DualKeyDict` の TODO 整理、定数化、命名の明確化。
- `api/pipeline.py`
  - [ ] `__repr__`、`cache(maxsize=env_default)`、エラーメッセージのガイド強化。

## 実施順（提案）
1. Quick Wins 全部（小 PR 複数でも良い）
2. Geometry digest 集約 + translation の委譲（小〜中 PR）
3. 形状キャッシュの一本化（中 PR、ADR 追記）
4. ルール整備（pyproject + pre-commit、ruff 警告の解消を段階導入）

---

## 期待効果
- 変換・キャッシュ・レジストリの“迷いどころ”が減り、コードを追う時間を短縮。
- ダイジェスト/キャッシュの責務が明確化され、バグクラス（ハッシュ不一致・二重キャッシュ）の芽を減らす。
- ルール化（pre-commit）で未使用インポート/変数の自然解消。レビュー負荷を軽減。

以上。優先度順に小さく進めれば、実装の体感シンプルさと読みやすさが着実に上がります。
