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
    - `docs/spec/repeat_effect_spec.md`（今回の仕様に同期）
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

- [ ] A-1. 3 フラグの意味を定義
  - `cumulative_scale: bool`
    - True: スケールはコピーごとに乗算で累積（幾何級数）。
    - False: コピー番号に応じて「始点 1.0 → 終点 scale」を線形補間して適用（1→scale の線形補間）。
  - `cumulative_offset: bool`
    - True: 平行移動は「前のコピーからの増分」として累積。
    - False: コピー番号に応じて「始点 0 → 終点 offset」を線形補間して適用  
      （インスタンス番号 `i` と `count` から `t = i / count` を求め、`offset_i = (1 - t) * 0 + t * offset` とする）。
  - `cumulative_rotate: bool`
    - True: 回転は「前のコピーからの増分」として累積。
    - False: コピー番号に応じて「始点 0 → 終点 angles_rad_step」を線形補間して適用  
      （インスタンス番号 `i` と `count` から `t = i / count` を求め、`angles_i = t * angles_rad_step` とする）。
- [ ] A-2. デフォルト値の方針を決定
  - 候補案 1: すべて True（現行の「完全累積」挙動に相当）。；すべて False でいいよ
  - 候補案 2: `cumulative_scale=True, cumulative_offset=False, cumulative_rotate=False`  
    （現行 `cumulative=False` のスケール挙動に近いプリセットをデフォルト化）。
  - ※どちらを採用するかユーザーと合意する。
- [ ] A-3. 既存の `cumulative: bool` 引数の扱いを決定
  - 案 a: 完全に廃止し、3 フラグに置き換える（シンプルだが破壊的）。；これを採用
  - 案 b: 非公開相当として残し、内部的に 3 フラグにマッピングする（互換性重視）。
  - 案 c: `cumulative` は deprecated としてサポートしつつ、推奨は 3 フラグに移行。
- [ ] A-4. 3 フラグ組合せごとの「よく使うプリセット」を整理；これはいらない。
  - 例:
    - 全累積: `(True, True, True)` → 反復変形・螺旋など。
    - 位置/角度のみ非累積: `(True, False, False)` → 等間隔配置 + スケール累積。
    - 完全非累積: `(False, False, False)` → 「1→scale の線形補間 + 等間隔配置 + 等間隔角度」。
- [ ] A-5. `docs/spec/repeat_effect_spec.md` との整合方針を決める；その md ファイルは無視していいよ。ややこしいからこちらで削除しておいた。
  - 現状の「cumulative 1 本軸 + scale モード」の記述を、3 フラグ前提に書き換える。

**要: ユーザー確認ポイント 1**

- 3 フラグのデフォルト値として「案 1」か「案 2」のどちらを採用するか。
- 既存の `cumulative` 引数を破壊的に廃止して良いか（案 a）か、互換レイヤを残すか（案 b/c）。

---

## B. 実装変更（引数・ロジック分岐の導入）

目的: `repeat` エフェクト本体に 3 フラグを導入し、モードごとの挙動を分かりやすく実装する。

- [ ] B-1. 関数シグネチャの更新案を確定
  - 例（案 a: `cumulative` 廃止）:
    - `def repeat(..., *, count: int = 3, cumulative_scale: bool = True, cumulative_offset: bool = True, cumulative_rotate: bool = True, ...)`
  - 例（案 b/c: 互換レイヤあり）:
    - `def repeat(..., *, count: int = 3, cumulative: bool | None = None, cumulative_scale: bool | None = None, cumulative_offset: bool | None = None, cumulative_rotate: bool | None = None, ...)`
    - `None` の場合は内部ポリシーに従って決定（`cumulative` が優先など）。
- [ ] B-2. 内部ロジックを「元座標 base」と「現在座標 current」に整理
  - `base_coords = coords.copy()` を基準とし、
    - offset/rotate の累積/非累積は「base を使うか current を使うか」で切り替える。
- [ ] B-3. スケール更新処理の分離
  - `cumulative_scale=True`:
    - 現行の `_update_scale(current_scale, scale_np)` を流用。
  - `cumulative_scale=False`:
    - コピー番号と `count` に応じたスケールベクトルを直接計算するヘルパを追加  
      （例: `compute_scale_factor(i, count, scale_np)`）。
- [ ] B-4. オフセット・回転の適用処理の分岐
  - `cumulative_offset` / `cumulative_rotate` に応じて:
    - 累積時: `current_coords` に対して「増分」を適用。
    - 非累積時: `base_coords` に対して「i 倍のステップ」を適用。
  - ループ内で `if` が増えすぎないように、必要ならモード別の小ヘルパ関数を作る。
- [ ] B-5. 早期リターン条件・lines 構築ロジックの再確認
  - `count <= 0` / `g.is_empty` / `offsets.size <= 1` の no-op 振る舞いが変わらないこと。
  - 「元の線 + コピー」の順序・本数を維持。
- [ ] B-6. numba JIT 対応
  - `_apply_transform_to_coords` / `_update_scale` のシグネチャ変更は最小限に留める。
  - フラグ判定は Python 側で完結させ、numba 関数には「最終的な scale / rotate / offset ベクトル」を渡す。

---

## C. docstring / `__param_meta__` / 仕様ドキュメントの更新

目的: 新しいフラグ構成が IDE / GUI / 仕様書から一貫して分かるようにする。

- [ ] C-1. `repeat` 関数の docstring 更新
  - Parameters に 3 フラグを追加。
  - 各フラグの意味とデフォルト（累積/非累積）の説明を 1〜2 行で記述。
- [ ] C-2. `repeat.__param_meta__` に 3 フラグを追加
  - 例: `"cumulative_scale": {"type": "bool"}` など。
  - 既存の `"cumulative"` エントリをどう扱うか（残す/削除/非推奨）を決めて反映。
- [ ] C-3. `docs/spec/repeat_effect_spec.md` の更新
  - 「cumulative 1 本軸」の説明を削除し、`cumulative_scale` / `cumulative_offset` / `cumulative_rotate` ベースの説明に書き換え。
  - 「よく使うプリセット例」を、3 フラグの組み合わせとして列挙。
- [ ] C-4. 必要であれば `repeat_effect_plan.md` 側にも「3 フラグ案に更新した」旨を追記。

---

## D. テスト追加・動作確認

目的: フラグ組み合わせごとの挙動が意図通りであり、既存ケースが壊れていないことを確認する。

- [ ] D-1. 基本ケースのテスト追加
  - `cumulative_scale=True/False` の違いを確認するテスト。
  - `cumulative_offset=True/False` の違いを確認するテスト。
  - `cumulative_rotate=True/False` の違いを確認するテスト。
- [ ] D-2. 代表プリセットのテスト
  - `(True, True, True)` / `(True, False, False)` / `(False, False, False)` など、代表的な組み合わせで座標が期待通りになること。
- [ ] D-3. no-op 条件のテスト
  - `count=0` / 空ジオメトリ / `offsets.size <= 1` のケースで入力コピーが返ること。
- [ ] D-4. Lint / Type / Test 実行
  - `ruff check --fix src/effects/repeat.py`
  - `mypy src/effects/repeat.py`
  - `pytest -q tests/effects/test_repeat_*.py`（実際のファイル名に合わせて実行）

---

## E. 互換性・フォローアップ検討

目的: 既存スケッチへの影響と、今後の拡張余地を整理しておく。

- [ ] E-1. 既存スケッチへの影響評価
  - リポ内での `repeat` の使用箇所を `rg` で洗い出し、3 フラグ導入後の挙動差を確認。
  - 必要であればサンプル更新やコメント追記。
- [ ] E-2. 互換レイヤ（`cumulative`）の扱い決定と実装
  - A-3 の方針に従い、必要であれば「旧 `cumulative` → 3 フラグ」のマッピングを実装。
- [ ] E-3. 将来の拡張メモ
  - 「コピー番号 `n` をエフェクト外に公開するか」「LFO 等との連携で使うか」など、今回の変更に紐づくアイデアがあれば追記。

---

## メモ / 質問候補

- デフォルト値はどれにするか:
  - すべて True で「従来 repeat の完全累積」挙動を保つか。
  - あるいは「よく使うモード」（例えば scale 累積のみ）に寄せた組み合わせを既定とするか。
- 既存の `cumulative` 引数の扱い:
  - 破壊的変更を前提に廃止してしまうか、移行期間として docstring に deprecate 明記のうえ残すか。
- 非累積スケールの具体ルール:
  - 本計画では「1→scale の線形補間」を想定しているが、他に使いたい形があればここで確定しておきたい。
