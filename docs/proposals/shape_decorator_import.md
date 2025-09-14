# @shape デコレータの公開インポート方針（提案）

日付: 2025-09-14
作成: pyxidraw6 チーム（提案）
対象: `src/api/shape_registry.py`, `src/shapes/registry.py`, `src/api/__init__.py`

## 背景と課題

- 現状、内部では `shapes.registry.shape` を用いて形状を登録している。
- 公開 API 側には `api/shape_registry.py` があるが、直感的なインポート経路（例: `from api import shape`）が
  デフォルトでは用意されておらず、ユーザーが独自 Shape を `@shape` で登録する際に
  「どこから import すべきか」が分かりにくい。
- `shapes.__init__` でも `shape` を再エクスポートしているが、`shapes/*` は原則内部実装層であり、
  一般ユーザーの依存先としては避けたい（将来の内部変更に弱い）。

## 目標（UX/設計原則）

- 単一で覚えやすい公開インポート経路を提供（迷わせない）。
- 内部実装（`shapes/*`）への依存を避ける。
- 既存コード・CI・スタブ（`api/__init__.pyi` 生成）への影響を最小化。
- `effects` 側と概念的に対称（将来の一貫性）。

## 選択肢と比較

1. `from api import shape`（推奨）

- 概要: API ルートで `shape` を再エクスポート。
- 利点: 最短・記憶しやすい・公開境界が明快・ドキュメントの負担が最小。
- 欠点: `api/__init__.py` に 1 行追加と `__all__` 更新が必要。型スタブに載せるかは要判断（任意）。

2. `from api.shape_registry import shape`（非推奨・廃止予定）

- 理由: 公開 API の単一路線を崩す冗長経路となるため。UX/学習コストの最小化に反する。

3. `from shapes.registry import shape`（非推奨）

- 理由: 内部層への直接依存となり、将来のリファクタリング耐性が低い。

4. `from api.plugins import shape`（将来構想として保留）

- 概要: プラグイン境界を明示する専用ネームスペースを新設。
- 利点: 拡張点の可視化が明確。
- 欠点: 新規 API/ドキュメント/テスト整備の追加コスト。現時点では過剰。

## 決定（破壊的変更）

- 公開の唯一のインポート経路として 1) `from api import shape` を採用する。
- 2. `from api.shape_registry import shape` は廃止（ドキュメントから削除、CI/テストも更新）。
- `register_shape` エイリアスは完全に削除（後方互換なし）。
- `shapes.__init__` 由来の `shape` は内部用途に限定し、ユーザー向けには案内しない。

## 使用例（ユーザー作成 Shape の登録）

```python
# 推奨: 短く覚えやすい経路
from api import shape
from shapes.base import BaseShape  # BaseShape は現状内部提供。ここは将来の再エクスポート検討余地あり。

@shape  # または @shape("custom_name") / @shape(name="custom_name")
class MyStar(BaseShape):
    def generate(self, *, spikes: int = 5, radius: float = 10.0, **params):
        ...  # Geometry を返す
```

補足:

- `BaseShape` の公開経路は現状 `shapes.base.BaseShape`。将来 `api.shapes` などで再エクスポートする選択肢も検討可能。
  

## 互換性と移行

- 本変更は後方互換を持たない（破壊的変更）。
- 影響: 既存の `from api.shape_registry import shape` または
  `from api.shape_registry import register_shape as shape` は実行時に ImportError となる。
- 推奨移行手順（機械的置換可）:
  - すべての `from api.shape_registry import shape` を `from api import shape` に置換。
  - すべての `register_shape` 使用箇所を `shape` に置換。
- バージョニング: `__api_version__` を +1（例: 4.0 → 5.0）に引き上げ、
  `__breaking_changes__` に本件を追記する。
- 型スタブ: `api/__init__.pyi` に `shape` を再エクスポートして IDE 補完の一貫性を確保する。

### アーキテクチャ文書の同期（必須）
- ルート `architecture.md` に「公開インポート経路は単一路線（`from api import shape`）」を明記する。
- 実装との差分が生じた場合、`architecture.md` を更新して同期を保つ（AGENTS.md の運用規約に準拠）。

## 実装メモ（この提案に基づく最小変更）

- `api/__init__.py`
  - `from .shape_registry import shape as shape` を追加。
  - `__all__` に `"shape"` を追加。
- `api/shape_registry.py`
  - `register_shape = shape` を削除（エイリアス廃止）。
  - `shape` シンボルを import/再輸出しない（完全破壊保証: `from api.shape_registry import shape` は不可）。
- `scripts/gen_g_stubs.py`
  - 生成する `api/__init__.pyi` に `shape` を再エクスポート（`shape: Any` で十分）することを必須化。
  - 固定 `__all__` に `"shape"` を追加（必須）。
- テスト
  - `tests/api/test_shape_import.py`（新規）:
    - `from api import shape` が import できる。
    - デコレータ経由の登録が `shapes.registry.list_shapes()` に反映される。
  - 既存テストのうち `api.shape_registry.register_shape` に依存するものを全面置換。
- ドキュメント
  - README/拡張ガイドから `api.shape_registry` 経路と `register_shape` 記述を削除し、`from api import shape` のみを記載。

## リスク/懸念

- ルート `api` が内部に触れる輸出点を増やすことになるが、1 行の再エクスポートで循環や重依存は発生しない見込み。
- `BaseShape` の公開経路は依然として内部指向。将来 `api.shapes` での最小再エクスポート（`BaseShape` のみ）を検討。
  本提案の適用範囲外だが、UX 一貫性の観点では次の候補: `from api import BaseShape`。

## 改善アクション・チェックリスト（要確認）

- [ ] `api/__init__.py` に `shape` を再エクスポート（推奨案 1 の実施）
- [ ] `__all__` に `"shape"` を追加
- [ ] ドキュメント更新（README/ガイドに import 例を反映）
- [ ] スモークテスト追加（`from api import shape` と登録/取得の動作確認）
- [ ] スタブ生成に `shape` を追加（必須）
- [ ] （任意・将来）`api` で `BaseShape` の薄い再エクスポートを検討
- [ ] （将来）`effect` も `from api import effect` を提供して対称性を確保（現状は据え置き）

## DoD（Definition of Done）

- [ ] `from api import shape` のみが動作し、旧経路・エイリアスは実際に不可（ImportError/属性非存在）であること。
- [ ] スタブを再生成し、スタブ同期テスト（`tests/stubs/*`）が緑であること。
- [ ] README・ガイド・サンプルをすべて置換済みであること。
- [ ] 変更ファイルに対する `ruff/black/isort/mypy` が通過していること。
- [ ] 既存テストのうち `register_shape` 使用箇所の全面置換後にテストが緑であること。

---

この方向性で問題なければチェックリストに沿って実装に進めます。
