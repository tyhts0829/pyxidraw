# 251101: Geometry メタデータ遅延実行（G/E 呼び出しの遅延化）検討

どこで: `sketch/251101.py`、API 層（`src/api/shapes.py`, `src/api/effects.py`）、中核（`src/engine/core/geometry.py`）
何を: G（shape）と E（effect。アフィンを含む）を“いずれも”遅延化し、`Geometry` には実体を持たせずメタデータ（shape 定義・effect 列）を積み上げる。実計算はレンダ/エクスポート/明示 realize 直前に一括評価する。
なぜ: バッチ最適化・キャッシュ効率・並列実行計画・UI 連携（Parameter GUI）の明快化による応答性と単純さの向上

関連コード参照

- `sketch/251101.py`:1 — 現状は `G.text()` で即時生成、`E.pipeline...` で実行して `Geometry` を返す
- `src/api/shapes.py`:1 — 現状は `G.<shape>(...) -> Geometry`（LRU キャッシュで即時生成）
- `src/api/effects.py`:300 — 現状は `Pipeline.__call__(g) -> Geometry`（量子化 →compiled→ 逐次適用＋終端/中間キャッシュ）
- `src/engine/core/geometry.py`:1 — 現状は `Geometry` が純粋配列＋即時アフィン実装（`translate/scale/rotate`）
- 仕様: `docs/spec/pipeline.md`（量子化/キャッシュ鍵・実行モデル）

## 追加要件と制約（ユーザー要望の反映）

- ユーザー定義の shape/effect（現行の拡張 API）を維持し、キャッシュ恩恵をそのまま受けられること。
- `Geometry` の `coords/offsets` を容易に取り出し、自由に数値計算へ利用できること（学習/実験用途の自由度確保）。
- 過去スケッチとの後方互換は不要（破壊的変更は可）。ただし API/設計は可能な限りシンプルに。

上記を満たすため、本検討では「ユーザー拡張の呼び出し契約を壊さない」「配列アクセスは簡単」を最優先とする。

## メリット

- 計算の合成・最適化

  - アフィン合成（将来候補）: `translate/scale/rotate` を 1 行列へ統合して一括適用する最適化は可能だが、現計画では未採用（シンプルさ優先）。
  - 冗長ステップの削除: no-op（パラメータが 0 等）を計画段階で除外。
  - ステップ順の安全な並べ替え（将来候補）: 可換な範囲の並べ替えは検討余地。現計画では順序固定。

- キャッシュ効率とキーの安定化

  - 形状/パイプライン定義の指紋（量子化済みパラメータ）から早期にキーを作れるため、評価前でもキャッシュ戦略を立てやすい。
  - 既存の終端/中間（prefix）キャッシュと併用し、再計算をより限定化可能。

- 並列実行計画の明確化

  - ジョブ DAG を構築し、評価点をまとめて Worker へ投入（`src/engine/runtime/worker.py`）。
  - 共有 prefix の合流で重複仕事を削減（同一メタ接頭辞の統合）。

- UI/Parameter GUI との親和性

  - 呼び出し時は「未指定＝ GUI に出る」パラメータだけを登録し、評価直前の時刻 `t` を注入して確定可能（`src/engine/ui/parameters/runtime.py`:1）。
  - パイプライン単位 UID（現行 `pipeline_uid`）でメタを追跡しやすい。

- 可観測性/デバッグ

  - 「実行プラン（形状 → 効果列 → 評価点）」の可視化が容易。HUD に統計/合成結果を表示しやすい。

- レイヤー/マテリアル指向の拡張に適合
  - `colorize` などのマーカー的エフェクトをメタ層で扱い、評価後にレイヤー分割描画へ展開しやすい（`docs/plans/251101_per_shape_style.md` との整合）。

## デメリット / リスク

- 実装/保守の複雑度増

  - ラッパ型 `LazyGeometry` を導入し、`base/plan/realize` を実装・維持する必要がある（方式を固定）。
  - effect/shape 本体のシグネチャは現状維持。遅延化は `@effect`/`@shape` デコレータで吸収するため、ユーザー実装の受け口変更は不要。ただしデコレータ適用漏れ検知と登録経路の一元化が必要。
  - 純粋性の厳格化が必要: 遅延の前提として副作用や外部状態依存を避ける規約を明文化し、lint/テストで担保する。

- 即時性/使い勝手の変化

  - `G.*()` も `E.pipeline(...)(g)` も即時に `Geometry` を返さず、`LazyGeometry` を返す。配列アクセスや明示 `realize()` で初めて実体化されるため、UX は「遅延評価前提」に変化する。
  - API 型/スタブ更新が必要（`G.* -> LazyGeometry`, `E.pipeline -> LazyGeometry`）。スタブ同期テストの更新も必要。
  - 後方互換は不要要件。必要であればヘルパ `realize(G(...))` のような薄い互換 API を別途提供可能。

- テスト/デバッグの難度上昇

  - 部分評価（フェンス）を用意しないと、途中結果の検証がしづらい。スナップショット系テストの更新が必要。

- メモリ使用量の増加

  - メタ鎖＋評価済みジオメトリの併存によりピーク増。中間キャッシュと二重化しない設計配慮が要る。

- 非可換/トポロジ依存効果への配慮

  - 本計画では評価順を固定し、途中の暗黙 realize や並べ替えは行わないため、特別なバリアは不要。`fill/partition/subdivide` 等も最終 realize 時に定義どおり順次適用する。

- 時間/cc 依存の凍結点

  - 遅延すると `t`/`cc` の値が評価時刻に引っ張られる。キャッシュ安定性のため、呼び出し時点で値（あるいは量子化済み値）を凍結・保存するルールが要る（Effects は「量子化後の値がそのまま実行引数」仕様との整合）。

- キャッシュ鍵の設計見直し
- 実行結果キャッシュは「内容ハッシュ `g.digest`」を廃止し、LazySignature（shape_spec + effect_chain + t/cc の量子化スナップショット）を第一級キーに統一する（破壊的変更）。
  - `src/api/effects.py` の結果キャッシュ鍵は `(geometry_digest, pipeline_key)` から `lazy_signature` へ変更する。コンパイル済みパイプラインの再利用鍵（effect_chain 署名）は従来方針を維持。

## ユーザー拡張（shape/effect）と配列アクセスの維持方針

- 形状（shape）の拡張

  - 登録・呼び出し契約は維持しつつ、`G.<name>(**params)` は `LazyGeometry` を返す（shape_spec を内部に保持）。
  - 量子化済みパラメータから shape_spec 署名を生成し、LazySignature に統合。必要に応じて shape 実体化結果のキャッシュ（結果共有）は realize フェーズで適用。

- エフェクト（effect）の拡張

  - 現行の `def effect(g: Geometry, **params) -> Geometry` 契約は維持（実行時インターフェース）。ただし呼び出し時は実行せず plan へ記録し、realize フェーズで初めて `Geometry` を渡して実行する。ユーザー側の実装変更は不要。
  - パイプライン鍵計算（実装バージョン＋量子化済み params）は従来どおりだが、LazySignature を第一級キーに昇格させ、実体化前から共有/バッチ化に利用する。

- 配列アクセス（自由度）の維持
  - `Geometry.as_arrays()`/`n_vertices`/`n_lines` 等の配列読み出し系 API は「必要に応じて自動 realize」してから返す。利用者は従来どおり `coords/offsets` を入手でき、自由に NumPy 計算へ利用可能。
  - 互換注記: 遅延効果（アフィンを含む）が未反映の内部配列ビューは露出しない。配列読み出し時は必要に応じて realize し、常に反映・整合済みの配列を返す。`copy=False` でも“反映後の読み取り専用ビュー”を返す設計を想定（必要なら `peek_raw()` のようなデバッグ専用 API を別途検討）。

## 設計オプション（導入パス）

- A) LazyGeometry ラッパ導入（互換性重視案・今回非必須）

  - `G.*()`/`E.pipeline(...)(g)` は `LazyGeometry` を返し、`realize()` で `Geometry` を得る。
  - 既存コード互換のため、描画/エクスポート直前に暗黙 `realize()` を挿入。
  - `Geometry` 本体は不変のまま維持（中核の単純さを保持）。

- B) `Geometry` にメタフィールド追加（侵襲的だがシンプル）

  - `__slots__` を拡張し、アフィン累積行列・エフェクト列・shape spec を格納。
  - `translate/scale/rotate` は配列計算せずにメタを更新。評価フェンス API（`realize()`）を追加。
  - 既存 API 呼び出しは変えず、戻りは `Geometry` のまま（中にメタ）。

- C) 段階導入（機能スライス別）

  - Phase 1: デコレータ導入＋`LazyGeometry` 基盤（shape/effect とも呼び出しは plan のみ）。
  - Phase 2: 評価バリア/部分 realize を導入し、非可換・トポロジ依存効果の安全を担保。
  - Phase 3: AABB 伝播・No-op スキップ・バッチ/共有など最適化を順次適用。

- D) デコレータ駆動の遅延化（shape/effect とも既定 lazy）
  - 仕組み: `@effect` デコレータで登録時にラップし、呼び出し時に受け取る `g` が Lazy なら「実行せずに (effect_name, params) を `g.plan` に追記して返す」。`g` が実体 `Geometry` の場合も即時実行はせず、`LazyGeometry.from_geometry(g)` によって Lazy に強制変換してから `plan` に追記する（セマンティクスを一貫）。
  - realize: パイプライン終端で `LazyGeometry.realize()` を 1 回実行し、`plan` を順序どおりに評価して `Geometry` を得る（評価順は固定のまま）。
  - ユーザー定義 effect も `@effect` で自動ラップされ、遅延経路へシームレスに移行できる。実評価時はラッパから元実装（`__effect_impl__`）を呼び出す。
  - shapes 側（@shape の活用）: `@shape` で登録時にラップし、`G.*(...)` は `LazyGeometry(base=shape_spec)` を返す（既定で lazy）。
  - 影響範囲: `effects/registry.py`（ラップ/実装参照保存）、`api/shapes.py`（Lazy 返却の分岐）、`api/effects.Pipeline.__call__`（出力が Lazy のとき realize する分岐）を中心に、変更は限定的。
  - バリア: 本方針では評価順を固定し、途中の暗黙 realize は行わない（最終段のみ）。したがって特別なバリアやフラグは不要。

## 推奨 API 変更（方針: 途中の暗黙 realize はしない）

1. すべて lazy ＋終端 realize（既定）

   - effect（アフィンを含む）と shape は登録時にデコレータでラップ。呼び出しは `LazyGeometry.plan`（effect）または `base=shape_spec`（shape）への登録のみ。パイプライン終端（レンダ/エクスポート/明示 realize）で一括評価。

2. `LazyGeometry` の導入

   - `base`（shape_spec または実体 `Geometry`）と `plan`（effect_chain）を保持。`as_arrays()/len()` 等の配列読み出し API またはパイプライン終端で realize。
   - 実体 `Geometry` を受け取った場合も `LazyGeometry.from_geometry(g)` で包み、遅延経路へ載せる（effects 側の実装を単純化）。

3. キャッシュ鍵の統一

   - 結果キャッシュは `LazySignature(shape_spec, effect_chain, t/cc)` を第一級キーにする（`g.digest` 非依存）。
   - コンパイル済みパイプラインの再利用鍵は従来の effect_chain 署名を継続。

4. 明示 realize API
   - `E.pipeline.realize()` を用意し、ユーザーが必要に応じて評価境界を明示できる。

## 仕様整合の留意点

- 量子化/署名（`docs/spec/pipeline.md`）

  - Effects: 量子化後の値を実行引数として用いる仕様を維持。遅延時も「呼び出し時点で量子化 → 凍結」を原則化。
  - Shapes: 呼び出し時に量子化済み params から shape_spec を生成し、LazySignature へ組み込む。実体配列は realize 時に生成。

- ランタイム/GUI（`src/engine/ui/parameters/runtime.py`:1）

  - `begin_frame()/set_inputs(t)` のタイミングで `t` を凍結し、登録時点の値をメタへ刻む。
  - `pipeline_uid` によりフレーム内の一意性を維持。

- キャッシュ（`src/api/effects.py`:300 付近）
  - 終端/中間キャッシュは評価時に適用。評価前から「shape/effect/t/cc を含む LazySignature」を一次キーとして採用し、`g.digest` 依存を排す設計へ移行する（本計画の前提）。

## 改定案: digest 廃止と署名キャッシュ（LazySignature）

- 方針

- 「内容ハッシュ（`Geometry.digest`）」を廃止し、「定義の署名（LazySignature）」をキャッシュの第一級キーにする。
  - `Geometry.digest` プロパティ／計算は提供しない（完全廃止）。
  - LazySignature は以下を量子化・直列化して blake2b-128 で要約。
    - shape_spec: `shape_name` + 量子化済み `shape_params`
    - effect_chain: 固定順序の `(effect_name, 量子化済み params)` の列（affine も通常の効果として含める）
    - 時間/入力: `t`/`cc` などランタイム注入値の量子化後スナップショット

- 利点

  - realize 前から安定キーでキャッシュ・DAG 共有・バッチングが可能。
- digest に依存しないため、遅延の効果を阻害しない。

- 注意/補足
  - 量子化ステップは既存仕様（`__param_meta__['step']` / 既定 1e-6）に準拠。
- `Geometry.digest` は廃止（検証/同一性判定も LazySignature または配列比較/専用 API に委譲）。
- 既存の `(geometry_digest, pipeline_key)` 設計は撤廃し、`(lazy_signature)` へ一括置換（破壊的変更）。

## 固定順序前提の効率化（Affine 以外）

- No-op/条件スキップ

  - `amplitude=0` の `displace`、`density=0` の `fill`、単位変換の `affine`、変化無しの `mirror` などを LazySignature 上で即座に無効化（実行プランから除去）。

- バウンディングボリューム伝播（AABB）

  - shape 生成時に AABB を持つ。AABB を静的に変換/拡張できる効果（例: アフィン、`displace(amplitude)`、`boldify(b)`）は、計画段階で上限見積もりを行い、粗判定と cull に利用する。

- バッチ/同種連結

  - 固定順序・同一パラメータの effect を持つ LazySignature 群を束ね、1 回の compiled 関数解決・ワーカー投入で複数 shape をまとめて処理（関数呼び出し回数・ディスパッチオーバーヘッドを削減）。

- 中間結果の共有（prefix 再利用）

  - 同一 prefix の LazySignature を検出し、中間結果を共有。既存の prefix LRU を「署名ベース」に拡張して、realize 前でも共有可能にする。

- ストリーミング/チャンク評価

  - 大規模ジオメトリは行単位/チャンク単位で realize し、メモリアロケーションのピークを抑制（最終結合は offsets 補正で線形に実行）。固定順序でも効果が線形合成可能な範囲で有効。

- レイヤー分割の先読み

  - `colorize` 等のメタを用い、レイヤー単位で LazySignature を分割・順序付けして描画。VBO/Uniform の切替回数最小化に寄与。

- 定数伝播・早期固定

  - LFO/cc をフレーム開始時に凍結し、量子化後の定数値として LazySignature に埋め込む。以降の比較・キャッシュ・バッチ判定を高速化。

- コスト対効果の所感
  - アフィンの並べ替え（可換性解析）よりも、上記の「No-op スキップ」「AABB 先行」「署名ベースのバッチ/共有」の方が実装コストに対する効果が高い。順序は固定のままで十分に恩恵を得られる。

## 影響範囲と互換

- 影響（主に内部実装・テスト）

  - `Geometry.as_arrays(copy=False)` は“反映後ビュー”を返すため、厳密には従来より「コピー発生の可能性」が増える（後方互換不要前提で許容）。
  - `tests/core/test_geometry*.py` の一部（ビュー性や微細な配列同一性を期待するケース）は更新が必要。
  - `effects/*` は契約維持（`Geometry` 入出力）。ユーザー定義 effect も変更不要。

- 互換（ユーザー拡張）
  - shape/effect の登録 API はそのまま。量子化規約は継続。キャッシュの内部鍵は LazySignature に変わるが、ユーザーコード側の変更は不要。

## 導入ステップ案（最小リスク順）

1. Phase 1（基盤導入）

   - `@shape`/`@effect` デコレータ導入。`G.*()` と `E.pipeline(...)(g)` は `LazyGeometry` を返し、呼び出し時は plan 登録のみ。
   - `LazyGeometry` 実装（`base`/`plan`/`realize()`）。パイプライン終端（レンダ/エクスポート/明示 realize）で一括評価。
   - ベンチ/検証：`tests/perf/test_pipeline_perf.py`, `tests/core/test_geometry*.py`。

2. Phase 2（評価バリア/部分評価）

   - `realize()`/`barrier()` ステップを導入し、非可換・トポロジ依存効果の順序安全を保証。
   - HUD に「評価境界/共有/バッチ」統計を追加。

3. Phase 3（最適化の適用）
   - No-op スキップ、AABB 伝播、prefix 共有、効果同士のバッチ化を段階実装。

## 判断材料（要約）

- 高い見返り: 大規模ジオメトリでの帯域削減、GUI 連携時の再計算最小化、DAG 化によるワーカ効率化。
- 明確なコスト: 評価フェンスの設計、キャッシュ鍵/凍結点の追加仕様、テスト更新負荷。

## 結論（提案 + 適用可否の意見）

- 今やるべきか（方針）
  - Option D（デコレータ駆動の全面遅延）: shape/effect とも既定で遅延。LazySignature と prefix 共有/バッチを併用すればメリットが大きい。全体として適用は十分現実的。
- 推奨: 両方 lazy（アフィンは E の一部として通常の effect と同列）。小さなステップで導入し、終端 realize に一本化。
 - 将来拡張: 評価フェンス/API メタ（effect 分類）を整備し、非可換・トポロジ依存効果の直前 realize を自動化。AABB 伝播・No-op スキップ・バッチ/共有を標準化する。

## API 契約（要点の明文化）

- `G.<name>(**params) -> LazyGeometry`（shape_spec を保持）。
- `E.pipeline(...)(g: Geometry | LazyGeometry) -> LazyGeometry`（effect_chain を累積）。
- `LazyGeometry.realize() -> Geometry`（明示評価）。`as_arrays()/len()` 等の読み出しでも必要時に自動 realize。
- 量子化は「float のみ量子化」（`__param_meta__['step']`／既定 1e-6、ベクトルは成分別）を踏襲。
- LazySignature は shape_spec と effect_chain、凍結した `t/cc` を元に生成。
