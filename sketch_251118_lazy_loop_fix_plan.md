# sketch/251118.py Lazy + 動的個数パターン対応 実装計画（案）

目的:  
`sketch/251118.py` のように `draw()` 内でループしながら `p(g)` をリストに積むパターンでも、

- LazyGeometry / Worker / Renderer はそのまま活かしつつ
- Parameter GUI（ParameterRuntime + SnapshotRuntime）から見たときに
  - 「ラベル付きパイプラインのパラメータが、CC に応じて増える全ポリゴンに一貫して適用される」
  - 「初期フレームの `N` に依存せず、GUI と実行パスの乖離が目立たない」

ように振る舞うようにする。

現状整理（詳細は `sketch_251118_lazy_loop_investigation.md` を参照）:

- Geometry/LazyGeometry/Worker/Renderer の挙動は仕様どおりで、ループ＋リスト返しでも描画自体は正しい。
- 問題は Parameter GUI 側:
  - 起動時に `ParameterRuntime` が **t=0, cc=0 の 1 フレームだけ** `user_draw` をトレースして Descriptor を作る。
  - その時点では `N=1` なので、1 個分のパイプライン（p0）のみが GUI に登録される。
  - 本番時に CC を上げて `N` を増やしても、新規パイプライン（p1..）には Descriptor/override が存在せず、常に実装デフォルト値で走る。
  - SnapshotRuntime は override を「登録済み Descriptor ID」にだけ当てる設計なので、新規パイプラインは GUI から見えない。

ここでは **ラベル付きパイプラインを「1 つの論理パイプライン」として扱い、パラメータを共有する** 方向で実装する案を立てる。

---

## A. 基本方針: Pipeline ラベルを GUI パラメータ ID の主キーにする

狙い:

- `E.label(uid="polygons").affine().fill()` のようにラベルを付けたパイプラインでは、
  - 初期フレームで観測された Descriptor を基準に GUI パラメータを定義し、
  - そのラベルを持つ後続パイプライン（CC に応じて増えた分）にも **同じパラメータ ID で override を適用**する。
- これにより:
  - `N` が CC に依存して変動しても、GUI のスライダーは常に「polygons パイプライン全体」のパラメータとして機能する。
  - Descriptor は 1 度だけ作ればよく、ランタイム中に ParameterRuntime を再トレースする必要はない。

制約:

- ラベル未指定のパイプラインは従来どおり「パイプライン UID ごと」にパラメータが分かれる（動的個数に効かせたい場合はラベル付与を推奨する）。
- ラベルが同じだが構造の異なるパイプラインを並べると GUI パラメータが共有される（ラベル付けの責任はスケッチ側にある前提）。

---

## B. 実装タスク（チェックリスト）

### B-1. Effect 用 Descriptor プレフィックスの仕様を整理する

- [ ] 現状仕様の整理
  - [ ] `ParameterContext.descriptor_prefix`（`src/engine/ui/parameters/value_resolver.py`）での prefix 生成を確認。
  - [ ] Effect の Descriptor ID・SnapshotRuntime の prefix がどう対応しているかを再確認する。
    - Runtime: `descriptor_prefix` → `"effect@{pipeline_uid}.{name}#{index}"` 形式。
    - Snapshot: `prefix = "effect@{pipeline_uid}.{name}#{step_index}"` 形式で override を検索。
- [ ] 新仕様案の決定
  - [ ] ラベル付きパイプラインの ID 形式を決める（例: `effect@label:{pipeline_label}.{name}#{index}`）。
  - [ ] ラベル未指定時は従来どおり pipeline_uid ベースにする（後方互換の簡易維持）。
  - [ ] 形状側（`scope="shape"`）の ID 形式は変更しない方針を確認する。

### B-2. 共通ヘルパ関数の導入

- [ ] 「Effect 用 Descriptor プレフィックス」を生成する共通関数を新設する。
  - [ ] 場所候補: `src/engine/ui/parameters/runtime.py` か `value_resolver.py`（ParameterContext 近辺）。
  - [ ] シグネチャ案:
    - `def make_effect_prefix(*, pipeline_uid: str, pipeline_label: str | None, effect_name: str, step_index: int) -> str: ...`
  - [ ] ラベル付きの場合と未指定の場合の出力を上記新仕様に沿って実装。

### B-3. ParameterRuntime / SnapshotRuntime / ValueResolver の統一

- [ ] `ParameterRuntime.before_effect_call` を修正し、prefix 生成に共通ヘルパを使う。
  - [ ] `ParameterContext` 生成部分はそのままにしつつ、`ParameterValueResolver` 側で prefix を差し替えるか、`ParameterContext.descriptor_prefix` の実装を内部的にヘルパへ委譲する。
- [ ] `ParameterValueResolver.resolve` 内で使っている `descriptor_id = f"{context.descriptor_prefix}.{key}"` を、新しい prefix 生成に追随させる。
  - [ ] ここでは `context.pipeline_label` を利用できるようにする（必要なら `ParameterContext` にヘルパ参照を持たせる）。
- [ ] `SnapshotRuntime.before_effect_call`（`src/engine/ui/parameters/snapshot.py`）を修正し、同じヘルパで prefix を構築する。
  - [ ] 既存の `if pipeline_uid: prefix = ... else: ...` 分岐を削除し、`make_effect_prefix(...)` に置き換える。
  - [ ] `pipeline_label` 引数を活用する（現実装では無視されている）。
- [ ] これにより:
  - [ ] ParameterRuntime が登録した Descriptor ID と SnapshotRuntime が override を照会する ID が完全一致することを確認する。
  - [ ] ラベル付きパイプラインでは `pipeline_uid` が異なっても同一 ID になることを確認する。

### B-4. ラベル付きパイプラインの UI/挙動確認

- [ ] `sketch/251118.py` を使って動作確認する（実行は手動で行う想定）。
  - [ ] `E.label(uid="polygons").affine().fill()` のパラメータ（例: `fill` の `density`）が GUI に 1 セットだけ表示されることを確認。
  - [ ] `cc[1]` を変化させて N（ポリゴン数）を増減させたとき、GUI から変更した `density` が **すべてのポリゴン**に一貫して適用されることを確認。
  - [ ] Parameter GUI 上で同名ラベルを持つ他のパイプラインが無いときに意図しない共有が発生しないことを確認。
- [ ] `sketch/251117.py` のような既存スケッチでも regress が無いかざっくり確認する。

### B-5. 永続化（overrides 保存形式）への影響整理

- [ ] Descriptor ID 形式を変えることによる影響を整理する。
  - [ ] 既存の `load_overrides` / `save_overrides`（`src/engine/ui/parameters/persistence.py`）が単に ID→値の辞書として扱っていることを確認。
  - [ ] 当リポジトリには外部ユーザーが居ない前提なので「互換性破壊 OK」という方針を再確認する。
- [ ] 必要であれば簡易的なマイグレーション案を検討（オプション）。
  - [ ] 例: 旧形式 `"effect@p0.fill#0.density"` を読み込んだ場合、ラベル付きパイプラインなら新形式へ写像する処理を `load_overrides` 側に入れるかどうか検討する。
  - [ ] ただし複雑化を避けるため、初期段階では「既存 overrides は一度リセットされる」仕様でもよいかどうかを判断する。

### B-6. ドキュメント更新

- [ ] `architecture.md` の Parameter GUI / Pipeline セクションに、ラベル付きパイプラインの新しい挙動を追記する。
  - [ ] 「Effect パラメータ ID は pipeline_uid ではなく（可能な限り） pipeline_label を主キーとして共有される」旨を簡潔に説明する。
  - [ ] 「動的に増えるパイプラインに GUI パラメータを適用したい場合は E.label(uid=...) を付ける」というガイドラインを明記する。
- [ ] 必要であれば `docs/` 配下（Parameter GUI 関連のドキュメント）があればそこも更新候補としてチェックする。
- [ ] `sketch_251118_lazy_loop_investigation.md` から本計画ファイルへのリンク/参照を追加する（必要に応じて）。

---

## C. オプション案（今回は採用しない方向）

今後の拡張候補として検討したが、現段階では複雑化を避けるため採用しない案:

- [ ] **ランタイム再トレース方式**:  
  ParameterRuntime をワーカー側でも動かし、動的に現れる shape/effect 呼び出しを随時 Descriptor に追加していく。
  - 利点: 動的トポロジ（N が CC/時間で変わるパターン）も GUI に完全追従できる。
  - 欠点: multiprocessing/pickle 制約や性能コストが増大し、設計が大きく複雑になる。
- [ ] **パイプライン毎に独立 GUI を持つ方式**:  
  ラベル付きでも pipeline_uid ごとに別 GUI を持ち、ユーザーが個別に操作できるようにする。
  - 本件（全ポリゴンに一括適用したい）とは目的が異なり、UI も煩雑になるため、現状のスケッチ用途には過剰と判断。

これらは必要になったタイミングで別途 ADR/計画ファイルを立てて検討する。

---

## D. この計画のゴール

- `sketch/251118.py` のような「ラベル付きパイプラインをループで量産する」スケッチにおいて:
  - Parameter GUI で定義された Effect パラメータが、CC に応じて増減する全ポリゴンに対して一貫して適用されること。
  - LazyGeometry と Pipeline キャッシュの設計を崩さず、実装の複雑さを最小限に抑えること。

このチェックリストで問題なければ、この方針に沿って実装し、完了した項目にチェックを入れながら進める。***
