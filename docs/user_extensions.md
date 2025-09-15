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

from api import shape  # 唯一の公開経路（破壊的変更後）
from shapes.base import BaseShape  # 現状は内部提供。将来 api 経由の再エクスポートを検討。
from engine.core.geometry import Geometry

@shape(name="MyStar")
class MyStar(BaseShape):
    def generate(self, *, points: int = 5, r: float = 50) -> Geometry:
        import numpy as np
        th = np.linspace(0, 2*np.pi, points*2, endpoint=False)
        rr = np.where(np.arange(points*2) % 2 == 0, r, r*0.5)
        xy = np.c_[rr*np.cos(th), rr*np.sin(th)]
        return Geometry.from_lines([xy])
```

- 名前は省略可能（`@shape` / `@shape()`）。クラス名から推論されます（`CamelCase → snake_case`）。
- 実体は `shapes.registry.shape` に委譲。

## 3) 使い方（登録後）

```python
from api import G, E

# 形状の生成
base = G.my_star(points=7, r=80)

# パイプラインに組み込み
pipe = (E.pipeline
          .ripple(amplitude=0.5)
          .my_fx()  # さきほど登録した effect を使用
          .build())

out = pipe(base)
```

## 注意
- ユーザ拡張は import 時にデコレータが実行されて登録されます。実行前に拡張モジュールを import してください。
- 既存の `effects.registry` / `shapes.registry` を直接 import しても動作しますが、今後の安定 API としては `api` 経由を推奨します。
- 依存方向のポリシー（architecture.md 参照）により、`engine/*` から registry を参照しない設計です。
