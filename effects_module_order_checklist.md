# エフェクトモジュール順序ルール適用チェックリスト

目的: `src/effects` 配下の各エフェクトモジュールに対して、記述順序ルール
（docstring → import → param_meta → エフェクト本体関数 → 本体関数への `__param_meta__` 属性追加 → ヘルパー関数群）
を適用したかどうかを管理する。

## モジュールごとのチェック

- [ ] `src/effects/affine.py`
- [ ] `src/effects/boldify.py`
- [ ] `src/effects/clip.py`
- [ ] `src/effects/collapse.py`
- [ ] `src/effects/dash.py`
- [ ] `src/effects/displace.py`
- [ ] `src/effects/drop.py`
- [ ] `src/effects/explode.py`
- [ ] `src/effects/extrude.py`
- [ ] `src/effects/fill.py`
- [ ] `src/effects/mirror.py`
- [ ] `src/effects/mirror3d.py`
- [ ] `src/effects/offset.py`
- [ ] `src/effects/partition.py`
- [ ] `src/effects/repeat.py`
- [ ] `src/effects/rotate.py`
- [ ] `src/effects/scale.py`
- [ ] `src/effects/style.py`
- [ ] `src/effects/subdivide.py`
- [ ] `src/effects/translate.py`
- [ ] `src/effects/trim.py`
- [ ] `src/effects/twist.py`
- [ ] `src/effects/weave.py`
- [ ] `src/effects/wobble.py`

