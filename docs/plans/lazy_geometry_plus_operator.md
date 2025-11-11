LazyGeometry の `+` 演算子対応 改善計画
=====================================

目的: LazyGeometry でも `Geometry` 同様に `+`（結合）を使えるようにし、遅延評価とキャッシュを活かして性能劣化を防ぐ。

ゴール / ノンゴール
------------------
- ゴール（今回の対象を限定）
  - `LazyGeometry.__add__` を実装し、「Lazy + Lazy」のみサポート。
  - 演算子は“即時実体化しない”設計（完全遅延）。
  - 多項結合のコストを O(totalN) に抑える（繰り返しコピーを避ける）。
  - 既存の Prefix LRU と両立し、キャッシュヒット率を下げない。
- ノンゴール（今回の範囲外）
  - `Lazy + Geometry`／`Geometry + Lazy` のサポート（非対応のまま）。
  - 幾何ブール演算（和/差/積）。
  - `__iadd__`（`+=`）。

期待仕様（挙動）
----------------
- 代数的性質（対象は「順序付きポリライン列」）
  - 結合順序は左→右を保持（左結合）。
  - 単位元は空 `Geometry`（`G.empty()`）。
  - 結果の `offsets` は後続の先頭を先行の頂点数だけシフトして結合（`Geometry.concat` と同義）。
- 組合せ
  - `Lazy + Lazy` → 遅延のまま返却（plan に「結合ステップ」を1つだけ追加/集約）。
  - `Lazy + Geometry` → 非対応（TypeError）。
  - `Geometry + Lazy` → 非対応（従来どおり失敗）。
  - `Geometry + Geometry` → 既存どおり（`Geometry.__add__`）。

設計方針（性能最優先）
----------------------
1) 遅延 step として実装（即時 `realize()` はしない）
- `engine.core.lazy_geometry` に新規エフェクト実装 `_fx_concat_many(g, *, rhs_sig_list, rhs_ref_list) -> Geometry` を追加。
- `__add__` は plan の末尾が `_fx_concat_many` なら右項をその `rhs_ref_list` に“集約”（フラット化）し、無ければ新規に1ステップ追加。
  - これにより `a + b + c + d` は「concat ステップ ×1」で表現でき、反復コピーを避ける。

2) 1パス結合（O(totalN））
- `_fx_concat_many` は以下で一括結合する:
  - `g` と `rhs_ref_list` の各要素（LazyGeometry）を必要時に `realize()`。
  - 総頂点数と総線本数を先に算出→出力配列を一括確保。
  - `np.vstack/hstack` の多段使用を避け、プレ割り当て配列へブロックコピー＋`offsets` シフトを一回で実施。

3) キャッシュ（Prefix LRU）と署名
- 署名づけ用に、右項ごとに `api.lazy_signature.lazy_signature_for(lg)` を用いて `rhs_sig_list` を作成。
- 実体参照は `common.types.ObjectRef` で保持（`rhs_ref_list`）。
  - 署名生成時は `rhs_ref_list` を“無視”し、`rhs_sig_list` のみを使用（詳細は次項）。

4) 署名生成時に実体参照を除外（キャッシュ劣化を防ぐ）
- `LazyGeometry.realize()` 内の「effect 署名生成」箇所を小変更し、`eparams` から以下を除外した dict を `params_signature()` に渡す:
  - キーに `_ref` を含む項目（例: `rhs_ref_list`）
  - または値が `ObjectRef` の項目
- これにより、キャッシュキーが ObjectRef（`id` 由来）に引きずられず、`rhs_sig_list` だけで安定化。
  - 既存エフェクトの挙動は不変（通常は ObjectRef をパラメータに含めないため）。

5) マルチプロセス互換（spawn 安全）
- `_fx_concat_many` はモジュールトップレベル関数として定義（ローカル関数/クロージャ禁止）。
- plan には「トップレベル関数参照＋シリアライズ可能な params(dict)」のみを積む。

6) メモリ制御の方針
- 大規模 `rhs_list` の場合でも、一括割り当ては 1 回のみ。
- `Geometry`/`LazyGeometry` は `as_arrays(copy=False)` で読み取り専用ビューを利用し、不要コピーを避ける。
- 事前に総量を見積り、一括確保＋逐次コピーでピークを単一配列に限定。

変更概要（実装タスク）
----------------------
- src/engine/core/lazy_geometry.py
  - [x] `class LazyGeometry` に `__add__` を追加（右項は LazyGeometry のみ受け付け）。
    - 末尾が `_fx_concat_many` の場合はそこへ右項を追記、無ければ新規ステップ追加。
  - [x] トップレベルに `_fx_concat_many(g, *, rhs_sig_list, rhs_ref_list)` を追加。
    - `rhs_ref_list` は `ObjectRef` の列（LazyGeometry の参照）。適用時に `unwrap().realize()` で実体化。
    - 1 パスで coords/offsets を構築。
  - [x] `realize()` の署名生成箇所を修正（`_ref` キーと `ObjectRef` 値を除外して計算）。

- architecture.md（LazyGeometry セクション）
  - [ ] 演算子方針と concat 集約の挙動を追記。

- テスト（tests/ 以下、新規）
  - [ ] `test_lazy_add_basic`: `Lazy+Lazy` の結果が `Geometry.concat` と一致。
  - [ ] `test_lazy_add_reject_mixed`: `Lazy+Geometry` と `Geometry+Lazy` は TypeError/非対応のまま。
  - [ ] `test_lazy_add_no_realize_on_add`: `__add__` 呼び出し時に `realize` が走らない（モックやフラグで検証）。
  - [ ] `test_lazy_add_chain_perf`: `a + b + ... + z` が O(totalN) で動作することをサイズ増加に対する計測で確認（smoke レベル）。
  - [ ] `test_picklable_plan`: `LazyGeometry` の plus チェーンが multiprocessing 経由でシリアライズ可能。

実装詳細ノート
----------------
- API 例
  - `g = G.grid(nx=10, ny=10)`（Lazy）
  - `h = G.circle(r=10)`（Lazy）
  - `out = g + h`（遅延のまま）。`len(out)` や `renderer` 適用時に初めて実体化。

- 署名パラメータ例（`_fx_concat_many`）
  - `eparams = {"rhs_sig_list": (b"sig_of_lg1", b"sig_of_lg2"), "rhs_ref_list": [ObjectRef(lg1), ObjectRef(lg2)]}`
  - 署名計算時は `rhs_sig_list` のみ使用し、`rhs_ref_list` は無視。

- 例外とバリデーション
  - `LazyGeometry.__add__(x)` の `x` がサポート外型なら `TypeError`。
  - 右項が空 `Geometry` の場合はステップ追加をスキップして早期返却。

パフォーマンス検証計画（編集ファイル優先）
------------------------------------------
- 変更箇所に限定して高速ループで確認
  - ruff/black/isort/mypy: 変更ファイルのみ。
  - pytest: 追加テストのみ、まずは `-m smoke`。
- 簡易マイクロベンチ（ローカル）
  - N=1e5 規模の頂点列を 10 本結合した場合の実行時間を `Geometry.concat` 逐次版と比較（目標: 同等かそれ以下）。

リスクと対策
------------
- リスク: 署名生成の変更が既存エフェクトに影響
  - 対策: 除外条件を「キーに `_ref` を含む」＋「値が `ObjectRef`」に限定（保守的）。
- リスク: 大量結合で一時メモリピーク
  - 対策: 事前に総量を見積り、一括確保＋逐次コピーでピークを単一配列に限定。
- リスク: 右項が巨大 Lazy の再実体化コスト
  - 対策: Prefix LRU のキーに `rhs_sig_list` を使うことで途中結果再利用を促進。

確認事項（要ご判断）
--------------------
- [x] `__iadd__`（`+=`）も同時に対応しますか？（今回は見送り前提）: No（見送り）
- [x] 署名計算での除外規則（`_ref` と `ObjectRef`）は上記方針で問題ありませんか？: Yes
- [x] 連結ステップの集約（フラット化）を常時有効とする前提で問題ありませんか？: Yes

受け入れ基準（DoD）
-------------------
- 追加 API が動作し、`LazyGeometry` で `+` が使用可能。
- `__add__` が実体化を発生させない（遅延維持）。
- チェーン結合で O(totalN) の振る舞い（テストで確認）。
- ruff/black/isort/mypy/pytest（変更・追加ファイル対象）が緑。
- architecture.md の更新が反映済み。

進め方（この計画に沿って）
--------------------------
1. 設計合意（本ファイルへのフィードバック反映）
2. 実装（`lazy_geometry.py` に限定）
3. テスト追加（最小セット）
4. ドキュメント更新
5. 変更ファイルに対する Lint/Type/Test 緑化 → 共有

補足提案
--------
- `Geometry.concat_many(list[Geometry]) -> Geometry` のユーティリティ追加で `_fx_concat_many` 実装を簡素化可能（任意）。
