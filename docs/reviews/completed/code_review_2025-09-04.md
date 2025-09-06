# PyxiDraw4 コードレビュー（2025-09-04）

- 対象: カレントリポジトリ（pyxidraw4）
- 主眼: 2025-09-03 の破壊的変更ガイドライン（Geometry 統一 / 関数エフェクト / E.pipeline / シリアライズと検証）への準拠、API一貫性、保守性、テスト整合性
- 結論（要約）:
  - 新アーキテクチャへの置換は概ね完了。`Geometry` 統一・関数エフェクト・`E.pipeline`・`to_spec/from_spec/validate_spec` の導入・テスト整備が行われている。
  - 一方で、いくつかの“不整合（旧パラメータ名の残存 / digest 無効化時の例外的挙動 / 軽微な重複）”が残っており、本番運用での回避不能エラーや動作差分につながるリスクがある。

---

## 破壊的変更ガイドラインへの整合性

- Geometry 統一: `engine/core/geometry.py` に単一 `Geometry`。`translate/scale/rotate/concat` は純関数（新インスタンス返却）で実装済み。
- エフェクト関数化: `effects/registry.effect` で `Geometry -> Geometry` 関数を登録。主要エフェクト（`rotation/translation/scaling/noise/filling/...`）は関数スタイルに移行済み。クラス継承は廃止。
- パイプライン一本化: `api/pipeline.py` の `E.pipeline ... .build()(g)` に統一。単層キャッシュ（LRU風・上限は `PXD_PIPELINE_CACHE_MAXSIZE`）あり。
- シリアライズ/検証: `to_spec(pipeline)`, `from_spec(spec)`, `validate_spec(spec)` 実装済み。`validate_spec` は登録名の存在/JSON整合性/関数シグネチャ検証/任意の `__param_meta__` も参照。
- プロジェクト構成: `api/`, `effects/`, `shapes/`, `engine/`, `benchmarks/`, `tests/` に整理。README/チュートリアル/テストも新API準拠。

総評: 方針は明確で、実装も全体に整っている。以降は“細部のスパイク”の解消を推奨。

---

## 重要な問題点（優先修正）

1) `displace` のパラメータ名が旧名のまま残存（実行時 TypeError の原因）
- 影響: 実行時に `TypeError: displace() got an unexpected keyword argument 'intensity'` 等で落ちる。
- 該当箇所:
  - `main.py` L35-L41 付近: `.displace(intensity=cc.get(5, 0.3))`
  - `simple.py` L14-L20 付近: `.displace(intensity=cc.get(2, 0.3), time=t)`
  - `benchmarks/plugins/serializable_targets.py` L108 付近: `.displace(intensity=intensity, frequency=frequency)`
- 期待仕様（ガイドライン準拠）:
  - `amplitude_mm`（旧 `intensity`）
  - `spatial_freq`（旧 `frequency`）
  - `t_sec`（旧 `time`）
- 修正例:
  ```diff
  - .displace(intensity=cc.get(5, 0.3))
  + .displace(amplitude_mm=cc.get(5, 0.3))

  - .displace(intensity=cc.get(2, 0.3), time=t)
  + .displace(amplitude_mm=cc.get(2, 0.3), t_sec=t)

  - .displace(intensity=intensity, frequency=frequency)
  + .displace(amplitude_mm=intensity, spatial_freq=frequency)
  ```

2) `translate` の引数名の不一致（ベンチマーク）
- 該当: `benchmarks/plugins/serializable_targets.py` の `translate(offset_x=..., offset_y=..., offset_z=...)`
- 期待仕様: 関数エフェクト `translation.translate(g, *, delta=(dx,dy,dz))`
- 修正例:
  ```diff
  - E.pipeline.translate(offset_x=float(tx), offset_y=float(ty), offset_z=float(tz))
  + E.pipeline.translate(delta=(float(tx), float(ty), float(tz)))
  ```

3) Geometry の digest 無効化時の挙動が不統一
- 事象: `engine/core/geometry.py` にて、
  - `scale()` と `rotate()` の「早期 return ブランチ」で `obj._compute_digest()` を直接呼び出し、環境変数 `PXD_DISABLE_GEOMETRY_DIGEST=1` を無視してダイジェストを計算している。
  - 他メソッドでは `_set_digest_if_enabled(obj)` を使用しており不統一。
- 影響: digest 無効化ベンチ時に余分な計算が走る。設計意図と乖離。
- 修正方針: 当該箇所を `_set_digest_if_enabled(obj)` に統一。

4) 軽微な重複/表記ゆれ
- `effects/translation.py`: `translate.__param_meta__` が二重代入。
- `api/__init__.py` の冒頭ドキュメント: `displace(intensity=0.3)` は旧名。`amplitude_mm` に更新推奨。

---

## 推奨改善（品質・保守性）

- 早期検証の強化: `Pipeline.__init__` 時に `inspect.signature` を用いたパラメータ名チェックを任意で有効化し、実行前に不正パラメータを検知（`validate_spec` 相当の一部を組み込み）。
- ドキュメントの同期性: README の「キャッシュ制御」に `PXD_PIPELINE_CACHE_MAXSIZE` と `PXD_DISABLE_GEOMETRY_DIGEST` を明記（形状キャッシュ向け `PXD_CACHE_*` とは別物である点を強調）。
- 互換性レイヤ（任意）: `effects.displace.displace` 内で旧キー（`intensity/frequency/time`）を読み替えて警告発行（ただし、関数署名の厳格性と `validate_spec` の方針を優先するなら現状維持が妥当）。
- `util.utils.load_config`: ファイル非存在時は `{}` を返す軽量ガードを追加（現状は呼び出し側で try/except 済みだが、ユーティリティ側で安全化すると再利用性が上がる）。

---

## テスト観点

- テスト整備は充実（`tests/` 直下の新API準拠テスト多数）。以下を追加検討:
  - `PXD_DISABLE_GEOMETRY_DIGEST=1` 時に `scale()`/`rotate()`（早期 return 条件を含む）がフォールバックハッシュでキャッシュ動作することの回帰テスト。
  - `benchmarks/plugins/serializable_targets.py` のシリアライズ互換ケースのスモークテスト（旧→新パラメータ名への移行を含む）。

---

## セキュリティ/設定

- `engine/io/cc/*.pkl`/`*.json` はデモ/サンプル用のポート名などを含む。秘匿性は低いが、運用プロジェクトでは差分肥大化を避けるために `.gitattributes` で LFS 化、または生成物を除外する運用も選択肢。
- `config.yaml` は安全な既定を提供。未存在時の挙動は呼び出し側でフェイルソフト（OK）。

---

## すぐに効く修正チェックリスト（優先順）

 - [x] `main.py` / `simple.py` / `benchmarks/plugins/serializable_targets.py` の `displace` と `translate` の引数名を新仕様へ更新。
 - [x] `engine/core/geometry.py` の digest 設定を `_set_digest_if_enabled()` に統一。
 - [x] `effects/translation.py` の重複 `__param_meta__` を整理。
 - [x] `api/__init__.py` のドキュメント例を新パラメータ名へ更新。
 - [x] README に `PXD_PIPELINE_CACHE_MAXSIZE` を追記。

---

## 参考（パッチ例）

> 本レビューはコード修正は行わず提案のみ。以下は適用イメージです。

```diff
# engine/core/geometry.py（抜粋）
- obj._digest = obj._compute_digest()
  _set_digest_if_enabled(obj)
```

```diff
# main.py（抜粋）
- .displace(intensity=cc.get(5, 0.3))
+ .displace(amplitude_mm=cc.get(5, 0.3))
```

```diff
# simple.py（抜粋）
- .displace(intensity=cc.get(2, 0.3), time=t)
+ .displace(amplitude_mm=cc.get(2, 0.3), t_sec=t)
```

```diff
# benchmarks/plugins/serializable_targets.py（抜粋）
- .translate(offset_x=float(tx), offset_y=float(ty), offset_z=float(tz))
+ .translate(delta=(float(tx), float(ty), float(tz)))

- .displace(intensity=intensity, frequency=frequency)
+ .displace(amplitude_mm=intensity, spatial_freq=frequency)
```

---

## ポジティブな所見

- アーキテクチャ: 新方針（関数エフェクト/統一 Geometry/パイプライン）に沿ったミニマルで見通しの良い設計。レジストリ（`common.base_registry`）で拡張容易。
- キャッシュ: 形状（`lru_cache`）とパイプライン（単層LRU）の二層で現実的。digest 無効化時のフォールバックも備える。
- テスト: API・幾何・パイプライン・検証・境界条件を幅広く網羅。CI への組み込みが容易。
- ドキュメント/チュートリアル: 実行パス（ヘッドレス含む）が明快で、新規利用者のオンボーディングに有用。

---

## 次の一手（任意）

- Pipeline の“ビルド時検証”フラグ（例: `E.pipeline.strict(True)`) を導入し、実行前にパラメータ不整合を検出。
- 仕様シリアライズに versioning（`spec_version`）を持たせ、将来の移行時に自動正規化（エイリアスの機械変換）を可能に。
- `effects` のパラメータ仕様（`__param_meta__`）から docs を自動生成（cheatsheet 更新を自動化）。

---

本レビューが、API の統一感と実運用の堅牢性向上に役立てば幸いです。必要であれば、上記パッチの適用やテスト追加も対応します。

---

## フォローアップアクション（今回対応）

- [x] README に `PXD_DISABLE_GEOMETRY_DIGEST` を追記し、`PXD_CACHE_*` / `PXD_PIPELINE_CACHE_MAXSIZE` / `PXD_DISABLE_GEOMETRY_DIGEST` の役割差分を明記。
- [x] `util/utils.load_config()` をフェイルソフト化（非存在/読込失敗で `{}` 返却）。
- [x] `engine/io/manager.py` が設定未存在時にも安全に動作するよう `get('midi_devices', [])` に変更。
- [x] `E.pipeline.strict(True)`（`PipelineBuilder.strict`）を実装し、ビルド時に未知キーを検出。
- [x] `effects.displace.displace` に旧キー（`intensity/frequency/time`）の互換受理＋ `DeprecationWarning` を追加。
 - [x] テスト追加：
  - パイプラインキャッシュが「digest 無効 + `rotate` 早期 return」および「empty + `scale` 早期 return」でも有効であること。
  - ベンチマークのシリアライズターゲットが `translate.delta` と `displace.amplitude_mm/spatial_freq` に正しくマップされること（レガシー名入力のスモーク）。
