# src/ 配下コードレビュー（2025-09-29）

## 総評
- 設計は明確で一貫。単一 Geometry、薄い API（G/E）、レジストリ/キャッシュ、UI パラメータ運用が整理され可読性も高い。
- ドキュメンテーションと型注釈が充実し、実装の意図と境界が分かりやすい。
- 主要懸念は「任意依存（numba/shapely）のトップレベル import」と「effects の一括 import」による環境依存の脆さ。

## 良い点
- 統一 Geometry 実装の堅牢さ（正規化・digest・読み取り専用ビュー・純関数変換）。
  - 例: `src/engine/core/geometry.py`
- API 境界の分離と拡張ポイント（生成=G, 加工=E, 描画=engine）。
  - 例: `src/api/__init__.py`, `src/shapes/registry.py`, `src/effects/registry.py`
- キャッシュ設計（G の LRU、Pipeline の量子化署名＋compiled LRU、HUD 連携）。
  - 例: `src/api/shapes.py`, `src/api/effects.py`
- パラメータ GUI 方針の順守（実値運用、RangeHint は __param_meta__ のみ、override は表示上）。
  - 例: `src/engine/ui/parameters/value_resolver.py`, `src/engine/ui/parameters/state.py`
- 遅延 import/フォールバック配慮（sketch の GL/MIDI、DPG ウィンドウ）。
  - 例: `src/api/sketch.py`, `src/engine/ui/parameters/dpg_window.py`

## 懸念点 / 改善ポイント
- 高: 任意依存のトップレベル import による失敗リスク（effects の一括 import が誘発）。
  - 一括 import: `src/effects/__init__.py`
  - numba トップレベル import の例: `src/effects/affine.py`, `src/effects/displace.py`, `src/effects/fill.py`, `src/effects/subdivide.py`, `src/effects/repeat.py`, `src/effects/weave.py`
  - shapely トップレベル import: `src/effects/offset.py`
  - 良い手本（存在時のみ使用・フォールバックあり）: `src/effects/dash.py`
- 中: compiled pipeline のグローバル LRU 上限が環境変数未設定時 None（無制限）。
  - 参照: `src/api/effects.py`
- 低: __param_meta__ の type 記法のばらつき（"number"/"integer"/"vec3" 等）。
  - 解釈は resolver 側で吸収済みだが、表記ガイドの明文化で可読性向上余地。

## 提案（実装方針の要点）
- 任意依存の import ガード統一
  - numba: `try: from numba import njit ... except: njit = lambda f: f` もしくは NumPy 経路フォールバックへ切替。
  - shapely: 関数内遅延 import に変更し、未導入時は明確なメッセージで例外/フォールバック。
  - effects/__init__: 個別の `try/except ImportError` で緩和（該当エフェクトのみ未登録扱い）。
- compiled pipeline の上限既定を導入（例: デフォルト 256、環境変数で上書き）。
- __param_meta__ 記述ガイドの軽整備（type は number/int/bool/vec3、choices の利用など）。

## 次アクション（チェックリスト：実施前に要確認）
- [ ] effects/__init__.py の一括 import を個別 `try/except` に変更（未導入依存を許容）。
- [ ] numba 依存の各エフェクトでガード統一（存在時のみ njit、無い場合は純 Python/NumPy 経路）。
- [ ] offset エフェクトの shapely 依存を関数内遅延 import ＋メッセージ整備。
- [ ] compiled pipeline グローバル LRU の既定上限（例: 256）を導入し、環境変数で上書き可能に。
- [ ] __param_meta__ の表記ガイドラインを docs に追記（最小の記述で可）。
- [ ] 軽量検証: 任意依存未導入環境で `from api import E` → `E.pipeline.rotate(...).build()` が通ることを確認。

## テスト/型/スタブ観点
- スタブ同期テストが存在（`tests/stubs/test_g_stub_sync.py`）。公開 API 変更時は stub 再生成が前提。
- mypy は段階導入（`pyproject.toml`）。現状の型注釈は十分で、段階拡大に耐え得る品質。
- effects import 緩和後は、該当エフェクト未登録ケースのテストを追加すると安心（ImportError ではなく「そのエフェクトが利用不可」であることを明示）。

## 補足
- 本レビューは「現状の振る舞いを保ったまま安全性と移植性を高める」観点を優先。
- 実装に入る場合は、このチェックリストをベースに段階的に適用し、各段で ruff/mypy/pytest を変更ファイル単位で回す想定です。

