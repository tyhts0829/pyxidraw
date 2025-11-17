# repeat エフェクト cumulative\_\* フラグ導入 改善計画

目的: `repeat` エフェクトに `cumulative_scale` / `cumulative_offset` / `cumulative_rotate`  
という 3 つの独立した bool スイッチを導入し、スケール・平行移動・回転の累積/非累積を  
個別に切り替えられるようにする。既存の挙動をできるだけ自然に包含しつつ、  
API/実装はシンプルで理解しやすい形を目指す。

本ファイルは「実装に着手する前の計画メモ」であり、このチェックリストに基づいて  
実装・テスト・ドキュメント更新を進める。着手前にここでの方針を要確認とする。

---

## スコープ

- 対象:
  - `src/effects/repeat.py`
  - テスト（新規）: `tests/effects/test_repeat_*.py`（名称は実装時に確定）
  - 仕様/ドキュメント:
    - `repeat.__param_meta__`
    - `repeat` 関数の docstring
- 非対象（今回やらないこと）:
  - `effects` 他モジュールの API/挙動変更
  - Parameter GUI のレイアウト大幅変更（テキスト/説明の更新にとどめる）
  - キャッシュ鍵・署名生成ロジックの変更

---

## チェックリスト（大項目）

- [ ] A. 仕様詳細の整理（各フラグの意味とデフォルト）
- [ ] B. 実装変更（引数・ロジック分岐の導入）
- [ ] C. docstring / `__param_meta__` / 仕様ドキュメントの更新
- [ ] D. テスト追加・動作確認
- [ ] E. 互換性・フォローアップ検討

以下で各項目を細分化する。

---

## A. 仕様詳細の整理（各フラグの意味とデフォルト）

目的: `cumulative_scale` / `cumulative_offset` / `cumulative_rotate` の意味と  
デフォルト値を明確化し、既存の `cumulative` パラメータとの関係を決める。  
さらに、「始点/終点は固定しつつ、どの程度“累積っぽく”変化させるか」を  
イージングカーブ（`t' = t**curve`）で制御する方針を整理する。

- [x] A-1. 3 フラグの意味を定義
  - `cumulative_scale: bool`
    - True: コピー番号 `i` から得られる `t = i / count` に対し、`t' = t**curve` を用いて  
      「始点 1.0 → 終点 scale」を非線形に補間して適用（curve > 1 で終盤に変化が偏る“累積感”のある挙動）。
    - False: コピー番号 `i` に応じて「始点 1.0 → 終点 scale」を線形補間して適用（`t = i / count` をそのまま使用）。
  - `cumulative_offset: bool`
    - True: `t = i / count` から `t' = t**curve` を求め、「始点 0 → 終点 offset」を非線形に補間して適用。
    - False: コピー番号に応じて「始点 0 → 終点 offset」を線形補間して適用  
      （インスタンス番号 `i` と `count` から `t = i / count` を求め、`offset_i = (1 - t) * 0 + t * offset` とする）。
  - `cumulative_rotate: bool`
    - True: `t = i / count` から `t' = t**curve` を求め、「始点 0 → 終点 angles_rad_step」を非線形に補間して適用。
    - False: コピー番号に応じて「始点 0 → 終点 angles_rad_step」を線形補間して適用  
      （インスタンス番号 `i` と `count` から `t = i / count` を求め、`angles_i = t * angles_rad_step` とする）。
- 補足: カーブパラメータ `curve: float`
  - 共通のスカラーとして導入し、`t = i / count` に対し `t' = t**curve` を計算する。
  - 既定値は `curve=1.0`（完全に線形）。`curve>1` で終盤に変化が集中、`curve<1` で序盤に変化が集中。
  - `cumulative_*` が False の場合は `curve` を無視し、常に `t' = t` として扱う。
- [x] A-2. デフォルト値の方針を決定
  - デフォルト値は `cumulative_scale=False, cumulative_offset=False, cumulative_rotate=False`（すべて False）とする。
  - 既定挙動は「始点/終点を線形に割るモード」（完全非累積）とし、累積感が欲しい場合は明示的に各フラグを True にする。
- [x] A-3. 既存の `cumulative: bool` 引数の扱いを決定
  - 既存の `cumulative` は完全に廃止し、3 フラグに置き換える（破壊的変更を許容）。
  - 互換レイヤとしての内部マッピングや deprecate 対応は行わない。

---

## B. 実装変更（引数・ロジック分岐の導入）

目的: `repeat` エフェクト本体に 3 フラグを導入し、モードごとの挙動を分かりやすく実装する。

- [x] B-1. 関数シグネチャの更新案を確定
  - `cumulative` は廃止し、3 フラグ + カーブパラメータで制御する。
  - 実装済みシグネチャ:
    - `def repeat(..., *, count: int = 3, cumulative_scale: bool = False, cumulative_offset: bool = False, cumulative_rotate: bool = False, offset: Vec3 = (0.0, 0.0, 0.0), angles_rad_step: Vec3 = (0.0, 0.0, 0.0), scale: Vec3 = (1.0, 1.0, 1.0), curve: float = 1.0, auto_center: bool = True, pivot: Vec3 = (0.0, 0.0, 0.0),) -> Geometry`
- [x] B-2. 内部ロジックを「コピー番号から決まる t/t'」中心に整理
  - 各コピー n（1..count）に対して `t = n / count` を計算し、フラグに応じて `t' = t` または `t' = t**curve_clamped` を用いる。
  - すべての変換は元座標 `coords` に対して一発で適用し、`current_coords` のような累積座標は使用しない。
- [x] B-3. スケール更新処理の分離
  - スケールは `base_scale = (1,1,1)` と `scale_np` の線形補間で決定し、フラグに応じて t/t' を使い分ける実装に変更。
- [x] B-4. オフセット・回転の適用処理の分岐
  - オフセット/回転も同様に「始点 0 → 終点値」の線形補間とし、フラグに応じて t/t' を使い分ける実装に変更。
- [x] B-5. 早期リターン条件・lines 構築ロジックの再確認
  - `count <= 0` / `g.is_empty` / `offsets.size <= 1` の no-op 振る舞いが変わらないこと。
  - 「元の線 + コピー」の順序・本数を維持。
- [x] B-6. numba JIT 対応
  - `_apply_transform_to_coords` はそのまま利用し、引数として「最終的な scale / rotate / offset ベクトル」を渡すのみとした。
  - 旧 `_update_scale` は不要となったため削除済み。

---

## C. docstring / `__param_meta__` / 仕様ドキュメントの更新

目的: 新しいフラグ構成が IDE / GUI / 仕様書から一貫して分かるようにする。

- [x] C-1. `repeat` 関数の docstring 更新
  - Parameters に 3 フラグと `curve` を追加し、線形補間/カーブ補間の説明を記述。
- [x] C-2. `repeat.__param_meta__` に 3 フラグと `curve` を追加
  - `"cumulative_scale"`, `"cumulative_offset"`, `"cumulative_rotate"` を `bool` として追加し、既存の `"cumulative"` エントリは削除済み。
  - `"curve"` を `float` として追加し、RangeHint を `min=0.1, max=5.0, step=0.1` として定義。
- [ ] C-3. 必要であれば `repeat_effect_plan.md` 側にも「3 フラグ案に更新した」旨を追記。

---

## D. テスト追加・動作確認

目的: フラグ組み合わせごとの挙動が意図通りであり、既存ケースが壊れていないことを確認する。

- [x] D-1. 基本ケースのテスト追加
  - 線形オフセット（`cumulative_offset=False`）を確認するテストを追加。
  - 線形スケール（`cumulative_scale=False`）を確認するテストを追加。
  - カーブ付きオフセット（`cumulative_offset=True, curve>1`）を確認するテストを追加。
  - Z 回転の線形補間（`cumulative_rotate=False`）を確認するテストを追加。
- [ ] D-2. 代表プリセットのテスト
  - `(True, True, True)` / `(True, False, False)` / `(False, False, False)` など、代表的な組み合わせで座標が期待通りになること。
- [x] D-3. no-op 条件のテスト
  - `count=0` について、入力コピーが返ることを確認（既存テストを維持）。
- [x] D-4. Lint / Type / Test 実行
  - `ruff check src/effects/repeat.py` を実行し、問題がないことを確認。
  - `mypy src/effects/repeat.py` を実行し、型エラーがないことを確認。
  - `pytest -q tests/effects/test_repeat_basic.py` を実行し、追加テストが通ることを確認。

---

## E. 互換性・フォローアップ検討

目的: 既存スケッチへの影響と、今後の拡張余地を整理しておく。

- [ ] E-1. 既存スケッチへの影響評価；これは不要
  - リポ内での `repeat` の使用箇所を `rg` で洗い出し、3 フラグ導入後の挙動差を確認。
  - 必要であればサンプル更新やコメント追記。
- [ ] E-2. 互換レイヤ（`cumulative`）の扱い確認
  - A-3 の方針に従い、「旧 `cumulative` 引数は廃止し、マッピングは実装しない」ことを最終確認。
- [ ] E-3. 将来の拡張メモ
  - 「コピー番号 `n` をエフェクト外に公開するか」「LFO 等との連携で使うか」など、今回の変更に紐づくアイデアがあれば追記。

---

## メモ / 質問候補

- 非累積スケールの具体ルール:
  - 本計画では「1→scale の線形補間」を想定しているが、他に使いたい形があればここで確定しておきたい。
