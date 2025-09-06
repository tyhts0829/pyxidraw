# 2025-09-06: Effect module file renames

目的: ファイル名と公開関数名の不一致を解消し、`effects.<module>.<function>` の整合性を確保しました。

影響範囲: 破壊的（import パスが変わります）。`E.pipeline` のエフェクト名は従来通り（関数名＝登録名）で変更ありません。

## 変更一覧（旧 → 新）

- `effects.translation` → `effects.translate`（`translate`）
- `effects.rotation` → `effects.rotate`（`rotate`）
- `effects.scaling` → `effects.scale`（`scale`）
- `effects.noise` → `effects.displace`（`displace`）
- `effects.filling` → `effects.fill`（`fill`）
- `effects.array` → `effects.repeat`（`repeat`）
- `effects.subdivision` → `effects.subdivide`（`subdivide`）
- `effects.buffer` → `effects.offset`（`offset`）
- `effects.transform` → `effects.affine`（`affine`）
- `effects.dashify` → `effects.dash`（`dash`）
- `effects.wave` → `effects.ripple`（`ripple`）
- `effects.trimming` → `effects.trim`（`trim`）
- `effects.webify` → `effects.weave`（`weave`）

（`boldify`, `collapse`, `explode`, `extrude`, `twist`, `wobble` は変更なし）

## マイグレーション例

```diff
- from effects.translation import translate
+ from effects.translate import translate

- from effects.rotation import rotate
+ from effects.rotate import rotate

- from effects.noise import displace
+ from effects.displace import displace
```

`E.pipeline` の利用はそのままです。

```python
from api import E
pipeline = (E.pipeline.rotate(angles_rad=(0, 0, 1.5708))
                      .displace(amplitude_mm=0.2)
                      .build())
```

## 備考

- `effects/__init__.py` は新ファイル名を import するよう更新済みです（レジストリ登録副作用の維持）。
- チュートリアル/ドキュメントの旧表記は順次更新しています。

