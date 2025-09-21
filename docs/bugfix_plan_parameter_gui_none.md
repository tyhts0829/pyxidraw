# Parameter GUI: None デフォルトでスライダーが例外になる不具合（affine/pivot 他）

更新日: 2025-09-21

## 概要 / 症状
- パラメータGUIの描画中に `TypeError: float() argument must be a string or a real number, not 'NoneType'` が発生し、ウィンドウ更新が止まる。
- 例外発生箇所: `src/engine/ui/parameters/panel.py:95`（スライダーの正規化で `float(None)` を呼んで失敗）。

## 直接原因（Where）
- スライダー描画で実値→比率(0..1)へ正規化する際、現在値が `None` のまま渡されて `float(None)` が呼ばれている。
  - 該当行: src/engine/ui/parameters/panel.py:95
  - 現在値の取得: src/engine/ui/parameters/panel.py:82（`store.current_value()` が `None` かつ `descriptor.default_value` も `None` の場合、そのまま `None` を返す）

## 根本原因（Why）
1) ベクトル型のメタを無視してスカラー扱いになる経路が存在
- `affine` の `pivot` は `Vec3 | None = None` で、メタは `"type": "vec3"`。
- しかし型判定 `_determine_value_type` が `"vec3"` を認識せず、`default` も実値も `None` の場合に最終的に `"float"` と誤判定される。
- その結果、`pivot` が「スカラー数値スライダー」として登録され、値が `None` のままスライダーに流れて例外。

2) 数値スカラーの `default=None` に対するフォールバックが未実装
- `offset.distance_mm: float | None = None` など、数値既定 `None` のパラメータも同様に初期描画で落ち得る。

## 影響範囲
- 旧APIでは `affine` 使用時（`pivot=None`）で高確率。
- 今後、`default=None` の数値/ベクトル型を持つ他のパラメータでも同様のクラッシュが起こり得る。

## 再現手順（例）
1) `main.py` で `affine` を使用し、パラメータGUIを有効化（既定設定）。
2) 起動直後の初回描画で `panel.py` 内で例外が出る。

## ベストな解決方針（設計）
「メタ→解決→GUI」の責務分離を崩さず、Resolver 側で正しい型分岐と既定値補完を行い、GUI 側は防御的に扱う。

1) メタ `"vec2"/"vec3"/"vec4"/"vector"` を `ValueType:"vector"` に正規化（Resolver）
2) 数値スカラーの `default=None` は RangeHint の中央値へ補完（Resolver）
3) GUI 層は保険として `None` を中央値へ退避（将来の回帰対策）
4) UX 改善（別案）: `affine` の `pivot=None` を廃止し、`auto_center: bool`（ON=平均座標）＋ `pivot: Vec3` に分離（本リポで採用済み）

## 実装チェックリスト（コード変更前の合意用）
- [ ] value_resolver: `_determine_value_type()` に `vec2/vec3/vec4/vector` → `"vector"` を追加。
- [ ] value_resolver: `resolve()` の分岐を meta 優先でベクトル経路へ。
- [ ] value_resolver: `_resolve_scalar()` で `default_actual is None` を RangeHint 中央値で補完。
- [ ] panel（安全策）: `SliderWidget` で `None` を中央値へ退避。
- [ ] テスト: Resolver 単体 + パネル描画の headless スモーク。
- [ ] ドキュメント: architecture.md に「Resolver は GUI に None を渡さない」方針を追記。

## 関連コード参照
- `src/engine/ui/parameters/panel.py:95`
- `src/engine/ui/parameters/value_resolver.py`
- `src/effects/affine.py`（`auto_center` 採用済み）

