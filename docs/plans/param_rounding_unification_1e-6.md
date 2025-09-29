# パラメータ丸め（量子化）1e-6 統一計画（提案）

目的
- effect と shape の引数のうち「浮動小数点」をキャッシュ鍵化する前段で 1e-6 に丸め（量子化）し、整数はそのまま（非量子化）とする。
- `__param_meta__['step']` が指定されていればそれを量子化に適用し、未指定の場合の既定は 1e-6（env 上書き可）。

背景 / 現状
- effect 側は `signature_tuple()` で量子化→ハッシュ可能化しており、既定は 1e-3（環境変数未設定時）。
  - 参照: src/api/effects.py:271, src/common/param_utils.py:175
- shape 側はランタイム有効時のみ `signature_tuple()` を使うが、ランタイム無効時は単なるハッシュ可能化で量子化していない。
  - 参照: src/api/shapes.py:214, src/api/shapes.py:220
- 量子化の既定ステップは `1e-3`（env 未指定時）。
  - 参照: src/common/param_utils.py:122

スコープ（やること）
- effect/shape の「浮動小数点」引数量子化ステップを既定 1e-6 に統一（ランタイム有無に依らず）。
- 整数（および bool）は非量子化でそのまま扱う。
- `__param_meta__['step']` があればそれを採用。ベクトルは成分ごとの `step` 指定にも対応（不足分は既定 1e-6）。
- 既存のアルゴリズム内の丸め（例: 辺数の整数化など）は非対象（そのまま）。

変更必要箇所（コード参照）
- src/common/param_utils.py:122
  - 量子化デフォルトの既定値を `1e-6` に変更（`PXD_PIPELINE_QUANT_STEP` 未設定時）。
- src/common/param_utils.py:132-135
  - `_quantize_scalar()` を「float（および np.floating）のみ量子化、int/bool はそのまま返す」に変更。
    - 例: `if isinstance(value, float) or isinstance(value, np.floating): ...`
- src/common/param_utils.py:157-171
  - ベクトル量（list/tuple）の各成分も float のみ量子化、int はそのままにすることをコメントで明示（実装は `_quantize_scalar` の更新で担保）。
- src/common/param_utils.py:175
  - `signature_tuple()` は `meta['step']` を尊重し、未指定成分の既定を 1e-6（env 上書き可）にする。
    - 実装案: `DEFAULT_SIGNATURE_STEP = 1e-6` を導入し、`quantize_params(params, meta, default_step=DEFAULT_SIGNATURE_STEP)` を呼ぶ。
- src/api/effects.py:271
  - 呼び出しは現状のまま（`signature_tuple` の内部仕様で float 量子化と `meta['step']` を適用）。
- src/api/shapes.py:214, src/api/shapes.py:220
  - ランタイム無効時も `signature_tuple()` を使用するよう統一（`_params_to_tuple` を廃止）。
  - 取得した `fn.__param_meta__` を渡す（`step` を量子化に適用）。

チェックリスト（実装手順）
- [x] 設計確定: 「整数は非量子化、浮動小数点のみ量子化」「既定は 1e-6、`__param_meta__['step']` を優先」を採用
- [x] 共通: 量子化デフォルト `1e-6` 化（env 未設定時の既定）
  - [x] `src/common/param_utils.py:122` の既定値を `1e-6` に変更
  - [x] 既定値は `_env_quant_step(None)` で一元化（専用定数は導入せず）
- [x] 共通: 量子化対象を「float のみ」に限定
  - [x] `src/common/param_utils.py:132-135` `_quantize_scalar()` を更新（int/bool はパススルー）
  - [x] ベクトル成分でも float のみ量子化（実装は `_quantize_scalar` で担保）。
- [x] 共通: API（署名生成）で `meta['step']` を適用
  - [x] `src/common/param_utils.py:175` `signature_tuple()` は現仕様で `meta['step']` を尊重
  - [x] 早期パス（変化なし参照返し）を `quantize_params` に追加
- [x] 共通: 署名一元化ヘルパーの導入と経路一本化
  - [x] `common/param_utils.py` に `params_signature(fn, params)` を追加
  - [x] `src/api/effects.py` と `src/api/shapes.py` で `params_signature` を使用
  - [x] `_params_to_tuple` を廃止し、参照箇所をすべて置換
- [x] effects: 呼び出し側は `params_signature()` を使用
  - [x] `src/api/effects.py:271` 近辺の鍵生成を `params_signature(fn, params)` に置換
  - [x] コメントを「float は 1e-6（または meta step）で量子化」に更新
- [x] shapes: ランタイム無効時も「float のみ 1e-6 量子化」で鍵生成
  - [x] `src/api/shapes.py:220` で `_params_to_tuple` → `params_signature` に変更
  - [x] `fn.__param_meta__` を取得して渡す（`step` を適用）
  - [x] コメント更新（ランタイム有無で鍵生成の方針差を解消）
- [ ] 影響調査: 旧挙動（1e-3/整数量子化）に依存した閾値テストや HUD 表示がないか確認
  - [ ] 影響があればテスト更新
- [ ] テスト追加（最小限）
  - [ ] `tests/common/test_param_utils_quantization.py`: float の既定 1e-6 と `meta['step']` 優先、int は不変を確認
  - [ ] `tests/api/test_shapes_cache_key_quantization.py`: ランタイム ON/OFF で float <step 差分は HIT、>=step は MISS／int 差分は MISS
  - [ ] `tests/api/test_effects_pipeline_quantization.py`: エフェクトでも同様の HIT/MISS 振る舞いを確認
- [ ] ドキュメント更新
  - [ ] `architecture.md`: キャッシュ鍵化の量子化仕様を「float のみ、`meta.step` 優先、既定 1e-6（env 上書き可）」に更新
  - [ ] `AGENTS.md`: Build/Test/Style/PR 最小項目に当該仕様の要点を追記
  - [ ] `docs/pipeline.md` や `docs/user_extensions.md` に量子化仕様の注意を追記
- [ ] 移行・周知
  - [ ] 変更はキャッシュキーと実行値（effects/shapes いずれも）に影響（float の量子化ステップが細かくなる or meta に従う）。
  - [ ] 既存キャッシュの互換性は不要（プロセス内キャッシュのみ）。必要なら再起動 or 明示クリアで解消。
  - [ ] `PXD_PIPELINE_QUANT_STEP` による上書きは継続（既定値は 1e-6）。

非対象（変更しないもの）
- アルゴリズム本体での丸め/整数化（例: `polygon(n_sides)` の整数丸め、`asemic_glyph` のスナップ角度など）。
- 画像エクスポートや HUD 表示など、キャッシュ鍵とは無関係の丸め。

受け入れ条件（DoD）
- 変更ファイルに対する `ruff/black/isort/mypy/pytest (-q -m smoke または対象テスト)` が成功。
- ランタイム有無で shape のキャッシュ鍵生成が統一され、float の <step 差分では CACHE=HIT、>=step では MISS、int の差分は常に MISS となること（tests で確認）。
- `architecture.md` と実装の説明が同期。

懸念・確認事項（要合意）
- [ ] 量子化の優先度は「`meta.step` > `PXD_PIPELINE_QUANT_STEP` > 既定 1e-6」で問題ないか。
- [ ] effect の実行時引数も量子化後の値（float は step 適用後）で渡す現行方針を維持して良いか（キャッシュ鍵と実行値の一貫性のため）。

改善提案（自然さ/直感性/コード量削減）
- 共通化ヘルパーの導入: `params_signature(fn, params) -> ParamsTuple` を `api/effects.py` と `api/shapes.py` の双方で使用し、`meta` の取得・量子化・タプル化を一元化。
- `_params_to_tuple` の廃止: shape 側も常に `signature_tuple` を使い、鍵生成経路を一本化（バグ温床を減らす）。
- 早期パス: `quantize_params` で「変化なし」の場合は元の参照を返す最適化（多数の int パラメータ時の無駄なタプル生成を削減）。
- 仕様の明文化: `architecture.md` に「int/bool は非量子化、float のみ step 適用、ベクトルは成分ごと」の規約を明示し、UI の RangeHint と混同しないよう注記。

参考（現状コードの該当行）
- src/common/param_utils.py:122
- src/common/param_utils.py:175
- src/api/effects.py:271
- src/api/shapes.py:214
- src/api/shapes.py:220

以上、問題なければこの計画に沿って実装に着手します。修正中に追加の確認事項が出た場合は本ファイルへ追記し、進捗チェックボックスを更新していきます。

---

検証結果（実行ログ抜粋）
- 変更ファイル限定 Lint/Format/TypeCheck
  - ruff/black/isort/mypy: OK（対象: `src/common/param_utils.py`, `src/api/effects.py`, `src/api/shapes.py`）
- スモークテスト: OK
  - コマンド: `pytest -q -m smoke`
  - 結果: 16 passed, 127 deselected in 約1.5s（macOS, Python 3.10）
