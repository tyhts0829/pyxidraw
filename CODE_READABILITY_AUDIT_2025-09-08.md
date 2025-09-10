# 可読性レビュー（2025-09-08）

このドキュメントは、リポジトリ全体を「可読性・素直さ・シンプルさ」の観点で走査した結果です。既存の厳しめレビュー（`CODE_REVIEW_2025-09-07.md`）は設計/互換/性能寄りの論点が中心だったため、本書では「読みやすさ・揃え方・命名・最小のコードで意図が伝わるか」に絞って指摘します。なお、変更は「必要十分」を旨とし、過度な抽象化や複雑化を避ける方針です。

---

## サマリー（まずはここから）
- 設計の一貫性とAPIの統一は良好で、ドキュメントも充実。可読性の大半は「表記の揃え」と「型/Docstringの一貫化」で改善できる。
- 優先度の高い“横断的な揃え”は以下の3点。
  1) Docstring/コメントの日本語統一（英語混在の削減）
  2) 型表記を PEP 585（組込みジェネリック：`list[...]/dict[...]`）へ統一
  3) 例外メッセージ/ログの文体と語彙の統一（簡潔・事実ベース）

---

## 横断改善（プロジェクト全体）

- Docstring/コメントの言語統一
  - 現状：日本語と英語の混在、および絵文字・装飾のばらつき。
  - 方針：Docstring/コメントは日本語で簡潔に。装飾（絵文字）はコード内では避け、`docs/`へ移動。
  - 代表例：
    - `src/common/logging.py`（英語Docstring）
    - `src/engine/render/renderer.py`（絵文字・装飾が濃い）
    - `src/shapes/sphere.py` のクラスDocstring（英語）

- 型ヒントの表記統一（Python 3.10）
  - 現状：`Dict/List/Tuple/Type` と `dict/list/tuple/type` が混在。
  - 方針：組込みジェネリック `dict[str, Any]` 等に統一。`typing` 由来は必要最小限（`Callable`, `Mapping`, `Sequence` など）。
  - 代表例：`src/common/base_registry.py`, `src/common/cacheable_base.py`, `src/util/constants.py`, `src/api/pipeline.py` ほか。

- 例外/ログ文面の統一
  - 現状：日本語/英語が混在。丁寧語/口語が混じる箇所もあり読み味が揺れる。
  - 方針：短く事実のみ（主語省略、終止形）。例：`"未登録: {name}"`, `"無効な引数: foo={v!r}"`。

- 命名の統一（pivot/center）
  - 現状：`Geometry` 側は `center`、一部エフェクトは `pivot`。
  - 提案：公開APIではどちらかに寄せる（候補：`pivot`）。内部では受け側で相互変換し、ユーザAPIと実装で乖離しないようにコメントで明示。

---

## モジュール別の具体指摘と小さな修正案

以下は「その場で直しても影響が小さく、読みやすさに効きやすい」順で記載します。パッチは“方向性のサンプル”です。

### 1) effects/registry.py — 型エイリアスとメッセージ
- 問題：`EffectFn = Callable[[Geometry], Geometry]` は実態（`**params` 受け）と乖離。
- 改善：`Callable[..., Geometry]` に変更し、未登録メッセージを日本語化。

```diff
-EffectFn = Callable[[Geometry], Geometry]
+EffectFn = Callable[..., Geometry]
@@
-def get_effect(name: str) -> Callable[..., Geometry]:
-    key = _normalize_key(name)
-    if key not in _REGISTRY:
-        raise KeyError(f"effect '{name}' is not registered")
+def get_effect(name: str) -> Callable[..., Geometry]:
+    key = _normalize_key(name)
+    if key not in _REGISTRY:
+        raise KeyError(f"未登録のエフェクトです: {name}")
     return _REGISTRY[key]
```

### 2) common/logging.py — Docstring の日本語化
- 問題：英語Docstring。方針とズレ。
- 改善：機能は現状維持で、説明のみ日本語に。

```diff
-"""
-Lightweight logging utilities for the project (Proposal 7).
-...略...
-"""
+"""
+軽量なロギング初期化ユーティリティ。
+
+アプリ側で未設定の場合のみ、最小限の設定を行う。既にハンドラがあれば何もしない。
+"""
```

### 3) api/pipeline.py — メッセージ/Docstringの揃え
- 問題：`validate_spec()` などで英語メッセージが混在。
- 改善：日本語化しつつ、文面を短く事実ベースに。

```diff
-raise TypeError(
-    f"spec[{i}] effect '{name}' has unknown params: {unknown}. Allowed: [{allowed_sorted}]"
-)
+raise TypeError(
+    f"spec[{i}] 不明なパラメータ: {unknown}（許可: [{allowed_sorted}]）"
+)
```

- 併せて Docstring の冒頭要約を 3–4 行に短縮し、詳細は `docs/pipeline.md` へ誘導（冗長さの削減）。

### 4) engine/pipeline/receiver.py — `__init__` の引数Docstring
- 問題：プライベート属性の説明が混ざり、引数説明として読み取りにくい。
- 改善：引数に焦点を当てて簡潔に。

```diff
     def __init__(self, double_buffer: SwapBuffer, result_q, max_packets_per_tick: int = 2):
-        """
-        _buffer:データを流し込む先のDoubleBuffer
-        _q (Queue): ワーカープロセスが作成したデータ（RenderPacket）が入るキュー
-        _max: 1回の更新(tick)で処理するパケットの最大数
-        _latest_frame: 最新のフレーム番号（古いデータを無視するため）
-        """
+        """受信キューから最新フレームを取り出して DoubleBuffer に流し込む。
+
+        引数:
+            double_buffer: ジオメトリを受け取る二重バッファ。
+            result_q: ワーカプロセスからの結果キュー（`RenderPacket` または例外）。
+            max_packets_per_tick: 1 tick で処理する最大パケット数。
+        """
```

### 5) shapes/sphere.py — Docstring とパラメータ変換
- 問題：クラス Docstring が英語。`subdivisions` 変換が手書きで重複。
- 改善：Docstring を日本語へ。変換は `common.param_utils.norm_to_int()` に寄せ、意図を明確化。

```diff
 @shape
 class Sphere(BaseShape):
-    """Sphere shape generator with multiple drawing styles."""
+    """複数スタイルに対応した球の生成器。"""
@@
-        MIN_SUBDIVISIONS = 0
-        MAX_SUBDIVISIONS = 5
-        subdivisions_int = int(subdivisions * MAX_SUBDIVISIONS)
-        if subdivisions_int < MIN_SUBDIVISIONS:
-            subdivisions_int = MIN_SUBDIVISIONS
-        if subdivisions_int > MAX_SUBDIVISIONS:
-            subdivisions_int = MAX_SUBDIVISIONS
+        from common.param_utils import norm_to_int
+        subdivisions_int = norm_to_int(subdivisions, lo=0, hi=5)
```

### 6) util/constants.py — 型表記の統一
- 問題：`Dict/Tuple/Union` を使用。
- 改善：組込みジェネリックへ寄せる（例）。

```diff
-from typing import Dict, List, Tuple, Union
+from typing import List

-CANVAS_SIZES: Dict[str, Tuple[int, int]] = {
+CANVAS_SIZES: dict[str, tuple[int, int]] = {
@@
-NOISE_CONST: Dict[str, Union[List[int], List[List[int]]]] = {
+NOISE_CONST: dict[str, list[int] | list[list[int]]] = {
@@
-PaperSize = Tuple[int, int]
-GradientVector = List[int]
+PaperSize = tuple[int, int]
+GradientVector = list[int]
```

### 7) engine/render/renderer.py — 過度な装飾の整理
- 問題：モジュール先頭 Docstring が長く装飾的で、関数Docstringより目立つ。
- 改善：要点を 3–4 行に縮約し、詳細は `docs/architecture.md` へ。コード中の説明は簡潔に。

```diff
-"""
-📌 全体の流れ（1フレームあたり）
-        1.	Rendererのtickが呼ばれる
-        ...（中略）...
-"""
+"""
+ライン描画レンダラ。DoubleBuffer の最新ジオメトリを GPU にアップロードし描画する。
+役割を Renderer/LineMesh/Shader に分離し、1フレームの処理を簡潔に保つ。
+詳細は docs/architecture.md を参照。
+"""
```

### 8) effects/offset.py — 変数名と補助関数の切り出し
- 問題：`vertices_list`/`new_vertices_list` などの同義語が多く、読み手が追いにくい。
- 改善：入力は `lines`, 出力は `out_lines` 等に統一。返り値/引数の型を明示（`list[np.ndarray]`）。
  Shapely 依存部は `_buffer_with_shapely(...)` に切り出し、条件分岐の意図を明確に（Shapely 1/2 対応は別チケット参照）。

---

## クイックチェックリスト
- [ ] Docstring/コメントの日本語統一（上記ファイルを優先）
- [x] PEP 585 の型表記へ統一（`dict/list/tuple`）
- [ ] 例外/ログの文面統一（短く事実のみ）
- [ ] `pivot`/`center` の用語統一（公開APIの語彙を先に決める）
- [ ] `effects/registry.py` の `EffectFn` 見直し
- [ ] `shapes/sphere.py` の `subdivisions` 変換を `norm_to_int` へ
- [ ] `engine/pipeline/receiver.py` の `__init__` Docstring 改善
- [ ] `engine/render/renderer.py` の先頭Docstring を整理

---

## 運用上の提案（読みやすさの維持）
- ルールを `ruff` で機械化
  - `pep8-naming`, `flake8-annotations` 相当のルールを有効化し、英語コメント比率/絵文字の使用を警告するプリカスタムを `pyproject.toml` に追加（必要最小限）。
- PR テンプレに「文体チェック」を追加
  - 例外メッセージと Docstring の言語確認を項目に追加。
- `docs/` との棲み分け
  - 長い背景説明や絵文字を含む図解は `docs/` に移し、コードの Docstring は 3–6 行の要約に。

---

## 備考
- 今回は“読みやすさ”に限定しており、性能/互換や CI 設定は範囲外です（必要なら別ドキュメントに分離）。
- 変更規模は小さく保ちやすいので、上から順に 1 ファイル 1 PR で進めるとレビュー負荷が低く、差分も読みやすいです。

以上。
