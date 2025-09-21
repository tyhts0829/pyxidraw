# affine: pivot=None を廃止し auto_center(bool) を導入する API 変更計画

更新日: 2025-09-21

## 背景 / 課題
- 現状の `affine(pivot: Vec3 | None = None, ...)` は、`pivot=None` のとき「ジオメトリの平均座標を中心にする」という暗黙挙動。
- GUI 側では `None` を直接扱えず、意味も可視化されにくい（ON/OFF の意図が伝わりにくい）。
- さらに、ParameterResolver/GUI の組み合わせによっては `None` がスライダーへ流入しクラッシュを誘発（詳細: docs/bugfix_plan_parameter_gui_none.md）。

## 目的（UX/設計）
- 「中心を使う/使わない」を明示のブールで切り替え可能にして、GUI で矛盾がない形にする。
- `None` に依存した挙動を排し、常に数値（実値）でパラメータを扱う。

## 提案する API 変更
- 旧: `affine(*, pivot: Vec3 | None = None, angles_rad: Vec3 = ..., scale: Vec3 = ...)`
- 新: `affine(*, auto_center: bool = True, pivot: Vec3 = (0.0, 0.0, 0.0), angles_rad: Vec3 = ..., scale: Vec3 = ...)`
  - `auto_center=True` のとき、ジオメトリの平均座標をピボットに採用。
  - `auto_center=False` のとき、`pivot` の数値を使用。

## UI への投影
- `auto_center`: bool → ToggleWidget（ON=中心を使用）。
- `pivot`: vec3 → 3 スライダー（`auto_center=False` の場合に意味を持つ）。
  - 現行 UI に無効化(disabled) 表現は無いので、ラベル注記で「auto_center=OFF で有効」を示す（将来のUI改善候補）。

## 互換性 / マイグレーション
- 破壊的変更（pivot=None の暗黙挙動を撤廃）。
- 置換ガイド:
  - 旧 `pivot=None` → 新 `auto_center=True`（`pivot` は未使用）。
  - 旧 `pivot=(x,y,z)` → 新 `auto_center=False, pivot=(x,y,z)`。
- 本リポは「破壊的変更可（未配布）」方針のため、内部修正で吸収可能。

## 実装チェックリスト（進捗）
- [x] effects/affine.py: シグネチャ変更（`auto_center: bool = True`, `pivot: Vec3 = (0,0,0)`）。
- [x] effects/affine.py: 実装変更（`auto_center` に応じて平均座標 or `pivot` を採用）。
- [x] effects/affine.py: 早期リターン条件の見直し（`angles`/`scale` が恒等なら中心に依存せず戻す）。
- [x] effects/affine.py: docstring 更新（振る舞いの明示化）。
- [x] effects/affine.py: `__param_meta__` に `auto_center`（bool）を追加。`pivot` は vec3 のまま。
- [x] 影響箇所の確認（呼び出し側は引数名ベースで互換、main.py は変更不要）。
- [x] architecture.md: affine のパラメータ仕様を更新（中心選択の明示化）。
- [x] 簡易テスト（smoke）:
  - [x] Case A: `auto_center=True` と `False` で結果が変わり得ること（180°回転で差分を確認）。
  - [x] Case B: `auto_center=False, pivot=(0,0,0)`, `angles=(0,0,0)`, `scale=(1,1,1)` で恒等出力。
- [x] 変更ファイルの限定チェック（ruff/black/isort/mypy, 対象テストのみ pytest）。

## 併走/関連（推奨だが別PRでも可）
- ParameterResolver 強化（vec2/vec3/vec4 を `ValueType:"vector"` と認識）
- 数値スカラーの `default=None` は RangeHint 中央値で補完（GUI安全化）
- 参考: docs/bugfix_plan_parameter_gui_none.md

## リスク / 緩和
- 破壊的変更: 既存呼び出し（内部）で `pivot=None` に依存していた場合、明示の `auto_center=True` へ置換が必要。
- UI 認知負荷: `pivot` のスライダーが常時表示される点は注記で緩和（将来、UIで無効化表示を検討）。

## DoD（完了条件）
- 新 API で `python main.py` が正常動作（GUI含む）。
- 上記チェックを通過し、関連ドキュメントが更新済み。

この計画で完了しています。追加の調整があれば追記します。

