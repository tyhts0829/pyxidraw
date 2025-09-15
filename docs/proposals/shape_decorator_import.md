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

- [x] `api/__init__.py` に `shape` を再エクスポート（推奨案 1 の実施）
- [x] `__all__` に `"shape"` を追加
- [x] ドキュメント更新（README/ガイドに import 例を反映）
- [x] スモークテスト追加（`from api import shape` と登録/取得の動作確認）
- [x] スタブ生成に `shape` を追加（必須）
- [ ] （任意・将来）`api` で `BaseShape` の薄い再エクスポートを検討
- [x] `effect` も `from api import effect` を提供して対称性を確保

## DoD（Definition of Done）

- [x] `from api import shape` のみが動作し、旧経路・エイリアスは実際に不可（ImportError/属性非存在）であること。
- [x] スタブを再生成し、スタブ同期テスト（`tests/stubs/*`）が緑であること。
- [x] README・ガイド・サンプルをすべて置換済みであること。
- [x] 変更ファイルに対する `ruff/black/isort/mypy` が通過していること。
- [x] 既存テストのうち `register_shape` 使用箇所の全面置換後にテストが緑であること。

---

この方向性で問題なければチェックリストに沿って実装に進めます。

## 付録: BaseShape 継承の必須解除（検討）

結論（TL;DR）
- 可。破壊的変更なしで、`@shape` が「関数」を受け取れるように拡張し、内部で `BaseShape` 派生の薄いラッパを自動生成すれば実現できる。
- 既存のクラスベース（`class Foo(BaseShape)`）はそのまま併存。ユーザーは「関数ベース（手早い）/クラスベース（高度）」を選べる。

狙いと UX
- 最短経路での拡張を提供: 「継承不要、関数一つで Shape 追加」
- `effect` と概念統一（関数登録）。
- 変換（center/scale/rotate）は従来どおり `BaseShape.__call__` で一元適用されるため、関数は“生成だけ”に集中できる。

最小仕様（提案）
- 受理対象: `@shape` 直下の「関数」または従来の `BaseShape` 派生クラス。
- 関数の戻り値: `Geometry` または「ポリライン配列のリスト」（既存互換）。
- 登録名: 既存と同じ正規化（Camel→snake、小文字化、`-`→`_`）。`@shape("custom")` も可。
- 変換: `G.foo(...).scale(...).translate(...)` などは従来どおり動作（ラッパが `BaseShape` を継承するため）。

実装概要（破壊なし）
1) `shapes.registry.shape` の拡張
   - もし引数 `obj` が「関数」なら、動的に `BaseShape` の薄いラッパクラスを生成して登録する。
   - 擬似コード:
     ```python
     if inspect.isfunction(obj):
         fn = obj
         class_name = _to_camel(resolved_name or fn.__name__)  # 表示用
         def _gen(self, **params):
             return fn(**params)
         _gen.__doc__ = fn.__doc__
         _gen.__signature__ = inspect.signature(fn)  # PEP 362: stub 生成のため
         Wrapper = type(class_name, (BaseShape,), {"generate": _gen, "__module__": fn.__module__})
         return _shape_registry.register(resolved_name)(Wrapper)
     ```
   - クラス受理パスは現状維持。

2) `api.shape_factory` 側の変更
   - なし。レジストリから返ってくるのは常に `BaseShape` 派生クラス（ラッパ含む）なので、既存ロジックのままで良い。

3) スタブ生成（`scripts/gen_g_stubs.py`）
   - 既存の「`shape_cls.generate` のシグネチャを読む」方式でそのまま機能。
   - 上記で `__signature__` をコピーするため、関数の引数が IDE 補完に反映される。

4) テスト
   - `@shape` で関数を登録→`G.<name>(...)` が動くこと、および変換が適用されることを追加で検証。

考慮点・リスク
- 状態を持つシェイプ（内部キャッシュや設定の切替）にはクラスのほうが適するため、「関数は手早い/無状態」というガイドを明記。
- `generate` 引数に `*args` を許さないポリシーは従来どおり（キーワード専用に正規化）。
- ラッパのクラス名は実行時に動的生成（デバッグ時の表示名用）。バイナリ互換には影響なし。

ドキュメント更新案
- ユーザガイドに「最短の関数ベース」「高度なクラスベース」の両パターンを掲載。
- 例（関数ベース）:
  ```python
  from api import shape, G
  from engine.core.geometry import Geometry
  import numpy as np

  @shape  # 継承不要
  def ring(*, r: float = 60.0, n: int = 200) -> Geometry:
      th = np.linspace(0, 2*np.pi, n, endpoint=False)
      xy = np.c_[r*np.cos(th), r*np.sin(th)]
      return Geometry.from_lines([xy])

  g = G.ring(r=80).translate(100, 100, 0)
  ```

導入規模と互換性
- 変更は `shapes.registry` のみ（デコレータ拡張）。既存 API/テストに破壊なし。
- 失敗時の例外ポリシー（TypeError / ValueError）は現状踏襲で問題なし。

チェックリスト（本件）
- [ ] `@shape` で関数を受理し、`BaseShape` ラッパを自動生成
- [ ] `generate.__signature__` の転写でスタブ品質を維持
- [ ] ユーザガイドに「関数ベース」の章を追加
- [ ] スモーク/単体テストを追加（登録→G→変換の一連）

---

## 代替案: 破壊的変更で「関数ベース」に全面統一（よりシンプル）

要約（Simplify 思想）
- `BaseShape` を公開 API から撤廃し、Shape も Effect と同様に「関数」を唯一の登録対象にする。
- `shapes.registry` は「関数のみ」を受け付け、`get_shape(name)` は関数を返す。
- `ShapeFactory (G)` はクラスを解決せず、関数を直接呼び出して `Geometry` を受け取る。

利点
- 実装・概念が Effect と完全対称になり、学習コストが下がる（“関数を登録して使う”だけ）。
- `BaseShape`・ラッパ生成・型境界の複雑さが消え、レジストリ/スタブ生成のコード量が減る。
- 既存の変換チェーン（`g.scale().translate()`）は `Geometry` 側にあるため、そのまま利用可能。

懸念/影響（互換）
- 破壊的: 既存の `class Foo(BaseShape)` 実装（内部/外部問わず）を関数へ移行する必要がある。
- もしクラスの状態や初期化順序に依存している Shape があれば、関数＋モジュール/クロージャのキャッシュに置換が必要。
- `BaseShape.__call__(center/scale/rotate, ...)` に依存する直呼びは使用不可に（ただし、公開 API `G` は従来から未使用）。

最小設計（Breaking 版）
- `shapes/registry.py`
  - 受け入れ対象: `Callable[..., Geometry | list[np.ndarray]]` のみ。
  - `get_shape(name)` は関数を返す。`list_shapes()`/`is_shape_registered()` は従来同様。
- `api/shape_factory.py`
  - `get_shape_generator(name)` は関数を返すように変更。
  - `instance.generate(**params)` をやめ、`fn(**params)` を直接呼ぶ。
  - 返り値が `Geometry` でなければ `Geometry.from_lines(...)` にフォールバック（現行ロジック踏襲）。
- `scripts/gen_g_stubs.py`
  - 形状の署名抽出を「クラスの `generate`」ではなく「関数本体」から取得するよう単純化。
  - これにより `__signature__` 転写などの小技が不要になり、実装が短くなる。
- `shapes/*` 実装
  - 代表例（Before）:
    ```python
    @shape
    class Polygon(BaseShape):
        def generate(self, n_sides: int = 3) -> Geometry: ...
    ```
  - 代表例（After）:
    ```python
    @shape
    def polygon(*, n_sides: int = 3) -> Geometry: ...
    ```

移行ガイド（外部ユーザー向け）
- 既存の独自 Shape クラスを関数に置換（クラスの `generate(**p)` の中身を関数本体へ移す）。
- 状態が必要な場合はモジュール変数や LRU キャッシュ（`functools.lru_cache`）で代替。
- 使用側は従来どおり: `g = G.polygon(...).scale(...).translate(...)`（変更なし）。

破壊的変更のスコープ/工数（見積）
- コード差分の主軸は 3 点: `shapes.registry`（簡素化）/`shape_factory`（関数呼び出し化）/`gen_g_stubs`（署名取得の単純化）。
- 既存の Shape 実装の大半は中身の移し替えのみで済む（特に無状態なもの）。
- 大型の `Text` などはモジュール内キャッシュが必要だが、既存でも内部で保持しているため影響軽微。

DoD（Breaking 版）
- [ ] `BaseShape` を公開 API から削除（内部にも残さない方針なら完全撤廃）。
- [ ] `shapes.registry` を「関数のみ」受理に変更（型とエラーメッセージ更新）。
- [ ] `api.shape_factory` を「関数呼び出し」に簡素化。
- [ ] スタブ生成（`gen_g_stubs`）を関数署名ベースに切替、同期テスト緑。
- [ ] 既存 `shapes/*` を段階的に関数へ移行（PR 分割可）。
- [ ] README/ガイド更新（“クラス継承は不要。関数だけ登録”を明記）。
- [ ] `__api_version__` を +1（例: 6.0）に引き上げ、`__breaking_changes__` に追記。

判断メモ
- 機能面は変わらず、UX/実装ともにスリムになるため、長期的には Breaking 版の方が保守性に優れる。
- 影響の大きさは「既存クラス Shape の数」に比例。段階移行（ラッパ併存→完全移行）も選択可能だが、今回の方針（すっきり重視）なら一気に進めるのが分かりやすい。
