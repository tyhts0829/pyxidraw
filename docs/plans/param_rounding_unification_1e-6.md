# パラメータ丸め（量子化）1e-6 統一計画（提案）

目的
- effect と shape の引数をキャッシュ鍵化する前段の丸め（量子化）を、プロジェクト全般で 1e-6 に統一する。
- `__param_meta__['step']` は UI の RangeHint のみで用い、数値の変換（量子化）には使わない方針へ合わせる。

背景 / 現状
- effect 側は `signature_tuple()` で量子化→ハッシュ可能化しており、既定は 1e-3（環境変数未設定時）。
  - 参照: src/api/effects.py:271, src/common/param_utils.py:175
- shape 側はランタイム有効時のみ `signature_tuple()` を使うが、ランタイム無効時は単なるハッシュ可能化で量子化していない。
  - 参照: src/api/shapes.py:214, src/api/shapes.py:220
- 量子化の既定ステップは `1e-3`（env 未指定時）。
  - 参照: src/common/param_utils.py:122

スコープ（やること）
- effect/shape の引数量子化ステップを 1e-6 に統一（ランタイム有無に依らず）。
- 量子化に `__param_meta__['step']` を使わず、デフォルト 1e-6 を用いる（UI RangeHint は別）。
- 既存のアルゴリズム内の丸め（例: 辺数の整数化など）は非対象（そのまま）。

変更必要箇所（コード参照）
- src/common/param_utils.py:122
  - 量子化デフォルトの既定値を `1e-6` に変更（`PXD_PIPELINE_QUANT_STEP` 未設定時）。
- src/common/param_utils.py:175
  - `signature_tuple()` の実装を更新し、量子化にメタの `step` を使わないようにする（固定で 1e-6）。
    - 実装案: `DEFAULT_SIGNATURE_STEP = 1e-6` を導入し、`quantize_params(..., default_step=DEFAULT_SIGNATURE_STEP)` を呼ぶ。
    - 併せて `quantize_params` に `use_meta_step: bool = False` を導入するか、`signature_tuple` 側で `meta={}` を渡して無視するかのどちらか。
- src/api/effects.py:271
  - `params_tuple = _signature_tuple(params, meta)` を、メタ `step` を無視した呼び出しに変更（`signature_tuple` 側の仕様変更で吸収）。
- src/api/shapes.py:214, src/api/shapes.py:220
  - ランタイム無効時も `signature_tuple()` を使用するよう統一（`_params_to_tuple` を廃止）。
  - 取得した `fn.__param_meta__` は渡すが、`signature_tuple` 側で `step` を見ない方針にする。

チェックリスト（実装手順）
- [ ] 設計確定: 「メタの step は UI RangeHint のみ、値変換には使わない」で合意を取る
- [ ] 共通: 量子化デフォルト `1e-6` 化（env 未設定時の既定）
  - [ ] `src/common/param_utils.py:122` の既定値を `1e-6` に変更
  - [ ] `DEFAULT_SIGNATURE_STEP = 1e-6` を導入し、ドキュメント化
- [ ] 共通: API（署名生成）でメタの step を無視
  - [ ] `src/common/param_utils.py:175` `signature_tuple()` を更新（`meta['step']` は無視）
  - [ ] 必要なら `quantize_params(..., use_meta_step=False)` を追加実装
  - [ ] 関連 docstring/コメントを更新
- [ ] effects: 呼び出し側は `signature_tuple()` の新方針に合わせる
  - [ ] `src/api/effects.py:271` 呼び出し部はそのまま（`signature_tuple` の内部仕様変更でカバー）
  - [ ] コメントを「1e-6 量子化に基づく署名生成」に更新
- [ ] shapes: ランタイム無効時も 1e-6 量子化で鍵生成
  - [ ] `src/api/shapes.py:220` で `_params_to_tuple` → `_signature_tuple` に変更
  - [ ] `fn.__param_meta__` を取得して渡す（ただし `step` は内部で無視）
  - [ ] コメント更新（ランタイム有無で鍵生成の方針差を解消）
- [ ] 影響調査: 旧挙動（1e-3）に依存した閾値テストや HUD 表示がないか確認
  - [ ] 影響があればテスト更新
- [ ] テスト追加（最小限）
  - [ ] `tests/common/test_param_utils_quantization.py`: 1e-6 での量子化境界の確認
  - [ ] `tests/api/test_shapes_cache_key_quantization.py`: ランタイム ON/OFF で <1e-6 差分は HIT、>=1e-6 差分は MISS
  - [ ] `tests/api/test_effects_pipeline_quantization.py`: エフェクトでも同様の HIT/MISS 振る舞いを確認
- [ ] ドキュメント更新
  - [ ] `architecture.md`: キャッシュ鍵化の量子化仕様を 1e-6 に更新し、`__param_meta__['step']` は RangeHintでのみ使用と明記
  - [ ] `AGENTS.md`: Build/Test/Style/PR 最小項目に当該仕様の要点を追記
  - [ ] `docs/pipeline.md` や `docs/user_extensions.md` に量子化仕様の注意を追記
- [ ] 移行・周知
  - [ ] 変更はキャッシュキーと引数実値（effects/shapes いずれも）に影響（量子化ステップが細かくなる方向）。
  - [ ] 既存キャッシュの互換性は不要（プロセス内キャッシュのみ）。必要なら再起動 or 明示クリアで解消。
  - [ ] `PXD_PIPELINE_QUANT_STEP` による上書きは継続（ただし既定値は 1e-6 に変更）。環境変数で従来 1e-3 に戻す運用も可能。

非対象（変更しないもの）
- アルゴリズム本体での丸め/整数化（例: `polygon(n_sides)` の整数丸め、`asemic_glyph` のスナップ角度など）。
- 画像エクスポートや HUD 表示など、キャッシュ鍵とは無関係の丸め。

受け入れ条件（DoD）
- 変更ファイルに対する `ruff/black/isort/mypy/pytest (-q -m smoke または対象テスト)` が成功。
- ランタイム有無で shape のキャッシュ鍵生成が統一され、<1e-6 の差分では CACHE=HIT となること（tests で確認）。
- `architecture.md` と実装の説明が同期。

懸念・確認事項（要合意）
- [ ] `__param_meta__['step']` を量子化ロジックで完全に無視して良いか（UI RangeHint のみに使用）。
- [ ] 量子化ステップの既定を 1e-6 に固定しつつ、`PXD_PIPELINE_QUANT_STEP` による上書きは継続で良いか。
- [ ] effect の実行時引数も量子化後の値（1e-6）で渡す現行方針を維持して良いか（キャッシュ鍵と実行値の一貫性のため）。

参考（現状コードの該当行）
- src/common/param_utils.py:122
- src/common/param_utils.py:175
- src/api/effects.py:271
- src/api/shapes.py:214
- src/api/shapes.py:220

以上、問題なければこの計画に沿って実装に着手します。修正中に追加の確認事項が出た場合は本ファイルへ追記し、進捗チェックボックスを更新していきます。

