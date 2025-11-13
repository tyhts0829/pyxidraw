# sketch/251110.py: draw の戻り値型警告（原因と改善計画）

## 症状
- Pylance 警告: 「型 `tuple[Geometry, Geometry]` は戻り値の型 `Geometry` に割り当てできません」
- 該当箇所: `sketch/251110.py:38` の `return p2(ring), p3(dot)`（関数定義は `def draw(t: float) -> Geometry`）

## 原因
- `draw()` の戻り値注釈が `Geometry` 単一である一方、実際には `tuple[Geometry, Geometry]` を返しているため静的型整合性が崩れている。
- 実行系は `Sequence[Geometry | LazyGeometry]` も受理可能（`src/api/sketch.py:112` 参照）だが、当該スケッチの注釈がそれに追随していないのが直接原因。

## 戻り値パターンの実態調査（現状仕様と実例）
- 受理仕様（ワーカ正規化）: `Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]`
  - 出典（関数引数型）: `src/api/sketch.py:112`
  - 正規化実装: `_normalize_to_layers(result)`（`src/engine/runtime/worker.py:76`）
    - Sequence の場合は各要素を `StyledLayer` 化（Lazy の `style` ステップを抽出）。
    - 単体 Lazy の場合も `style` があれば 1 レイヤー化。なければ Lazy/Geometry をそのまま返却。

- リポ内の draw 実装注釈と実例:
  - 単体 Geometry を返す: `main.py:11`, `demo/01.py:11`, `sketch/251101.py:15` ほか
  - Geometry | LazyGeometry を返す（Lazy を許容）: `sketch/251111.py:16`, `sketch/251113.py:16`
  - Sequence（レイヤー）を返す: 現状サンプル注釈は無しだが、設計上は許容（本件 `sketch/251110.py:38` は実質 `tuple[Geometry, Geometry]` でこの形に該当）

- 非対応/注意点:
  - `Iterable`/ジェネレータは Sequence 判定に通らないため非推奨（`list`/`tuple` を返す）。
  - ネストした Sequence（入れ子）は想定外。要素は `Geometry | LazyGeometry` の一次元に限定。
  - `None` を返す設計は無し（空を表したい場合は空 `Geometry` または空の `[]`）。
  - `StyledLayer` を直接返す API ではない（レイヤー化はワーカ側で行う）。

## 改善方針（2案）
どちらも警告は解消する。用途に応じて選択。

1) 単一 Geometry に結合して返す（推奨: シンプル）
   - 変更内容: `return p2(ring) + p3(dot)` に変更（`Geometry.__add__` は `concat` の糖衣: `src/engine/core/geometry.py:369`）。
   - メリット: 戻り値注釈 `-> Geometry` のままで整合。型が最も単純で保守容易。
   - デメリット: 将来レイヤーごとに別スタイルを適用したい場合は不向き（単一レイヤー化される）。

2) レイヤー出力として Sequence を返す設計に合わせる
   - 変更内容: 関数注釈を `def draw(t: float) -> Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]:` に変更し、現状の `return p2(ring), p3(dot)` をそのまま利用。
   - メリット: 将来のスタイル差分（色/太さなど）や HUD レイヤーの使い分けに拡張しやすい。
   - デメリット: 注釈が長くなり、Parameter GUI 周辺（`ParameterManager` は `Callable[[float], Geometry]` を受ける）との型注釈上の整合は `cast` に依存する状態が続く。
   - 補足: `run_sketch()` 内で `ParameterManager(cast(Callable[[float], Geometry], user_draw))` の扱いがあり（`src/api/sketch.py:186`）、ユニオン注釈でも実行パスは整合。

## 提案（初手）
- 現段階では 1) を採用し、戻り値を単一 `Geometry` に結合するのが簡明で安全。
- 将来的に個別スタイルやレイヤー制御が必要になったタイミングで 2) に移行するのが良い。

## 変更チェックリスト（実装前確認用）
- [ ] 方針を選択（推奨は 1）
- [ ] `sketch/251110.py` の `return` を選択方針に合わせて修正
  - 案1: `return p2(ring) + p3(dot)`
  - 案2: `def draw(t: float) -> Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]`
    - 必要なら import を追加: `from typing import Sequence`; `from engine.core.lazy_geometry import LazyGeometry`
- [ ] 変更ファイル限定の高速チェックを実行
  - Lint: `ruff check --fix sketch/251110.py`
  - Format: `black sketch/251110.py && isort sketch/251110.py`
  - Type: `mypy sketch/251110.py`
- [ ] 実行目視（任意）: `python sketch/251110.py` または `python main.py` から当該スケッチを呼び出し

## 影響範囲
- 当該スケッチファイルのみ。ランタイム/API 側の対応は既に整備済み（`Sequence` 受理）。

## 備考
- Parameter GUI を有効化していても、本件は型注釈上の警告であり実行は可能。
- レイヤー単位の別スタイル適用を行う予定があれば 2) を選択してください。

## 最終的な推奨注釈（案2採用時）
`sketch/251110.py:12` を次のようにするのが網羅的で一貫:

```
from typing import Sequence
from engine.core.lazy_geometry import LazyGeometry

def draw(t: float) -> Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]:
    ...
```

理由:
- ランタイムの受理仕様と厳密一致（`src/api/sketch.py:112`）。
- リポ内の既存実例（Lazy を返すスケッチ: `sketch/251111.py:16`, `sketch/251113.py:16`）も包含。
- `Sequence` を使うことで `tuple`/`list` を統一的に許容しつつ、ジェネレータ等の不安定な戻り値を避けられる。
