# ユーザ拡張ガイド（Shapes / Effects の登録）

外部（あなたのスケッチ/アプリ）から、独自の Shape や Effect を登録するための最小ガイド。
原則として api からのみ import すれば済むように、薄い委譲 API を提供しています。

---

## 1) Effect を追加（推奨エントリ）

```python
from __future__ import annotations

from api import effect  # 将来: ルート公開を予定（現状は effects.registry.effect を直接使っても可）
from engine.core.geometry import Geometry  # 型ヒント用（任意）


@effect(name="my_fx")
def my_fx(g: Geometry) -> Geometry:
    # ここで g を加工して新しい Geometry を返す
    return g
```

- 名前は省略可能（`@effect` / `@effect()`）。省略時は関数名から推論。
- 実体は `effects.registry.effect` に委譲。api から import できるため、依存の表面が安定します。

## 2) Shape を追加（推奨エントリ）

```python
from __future__ import annotations

from api import shape  # 唯一の公開経路
from engine.core.geometry import Geometry


@shape(name="my_star")
def my_star(*, points: int = 5, r: float = 50, inner: float = 0.5) -> Geometry:
    import numpy as np
    n = points * 2
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    rr = np.where(np.arange(n) % 2 == 0, r, r*inner)
    xy = np.c_[rr*np.cos(th), rr*np.sin(th)]
    return Geometry.from_lines([xy])
```

- 名前は省略可能（`@shape` / `@shape()`）。関数名から推論されます（`snake_case`）。
- 実体は `shapes.registry.shape` に委譲（api 経由で利用可能）。

## 3) 使い方（登録後）

```python
from api import E, G

# 形状の生成
base = G.my_star(points=7, r=80)

# パイプラインに組み込み
pipe = (E.pipeline
          .wobble(amplitude=0.5)
          .my_fx()  # さきほど登録した effect を使用
          .build())

out = pipe(base)
```

## 注意
- ユーザ拡張は import 時にデコレータが実行されて登録されます。実行前に拡張モジュールを import してください。
- 既存の `effects.registry` / `shapes.registry` を直接 import しても動作しますが、今後の安定 API としては `api` 経由を推奨します。
- 依存方向のポリシー（architecture.md 参照）により、`engine/*` から registry を参照しない設計です。

---

## 付録: `__param_meta__` の `step` の意味（量子化/RangeHint）

- RangeHint（UI の表示レンジ）は `min/max/step` を参照します（UI 側でのクランプ/表示のため）。
- キャッシュ鍵（署名）生成でも `step` を用います（`common.param_utils.params_signature`）。
  - 対象は float（`float | np.floating`）のみ。int/bool は非量子化。
  - 既定粒度は 1e-6。環境変数 `PXD_PIPELINE_QUANT_STEP` で上書き可。
  - ベクトルは成分ごとに適用。`step` をタプルや配列で与えると成分ごとに異なる粒度を指定できます（不足分は末尾値で補完）。
- 実行時の値
  - Effects: 量子化後の値が関数の実引数として渡されます。
  - Shapes: 量子化はキャッシュ鍵生成のみに使われ、関数にはランタイム解決後の値が渡されます。
