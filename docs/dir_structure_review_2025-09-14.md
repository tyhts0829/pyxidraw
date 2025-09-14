# ディレクトリ構成レビュー（初版）

- 日付: 2025-09-14
- 対象: 現行ツリー（api / engine / effects / shapes / common / util / tests / docs 他）
- 目的: 可読性と導線の明確化。vibe 開発時の迷いどころを減らす。

---

## 総評（要約）
- 大枠は明快で理解しやすい（api → engine → effects/shapes → common/util の層）。
- テスト/ドキュメントも領域で整理されており把握しやすい。
- 一方で、いくつかの“迷いの種”が散見され、ドキュメントの一貫性を強化するとさらに良くなる。

## 明快な点
- 層分離がディレクトリ構成に直結（`engine/core` / `engine/render` / `engine/pipeline` / `engine/ui` / `engine/io` / `engine/monitor`）。
- 公開 API が `api/` に集約（`G`/`E`/`run_sketch`）。
- `effects`/`shapes` はレジストリとデコレータで拡張一貫性あり。スタブ同期テストで公開面が守られている。

## 気になった点（混乱の芽）
- `api/shape_factory.py` と `api/shape_registry.py` が役割近接で二重窓口に見える（利用者は `G` で十分）。
- `common/` と `util/` の境界が名称だけだと直感しづらい（どちらも“共通”に見える）。
- ルート直下の `previous_code/` が現行と混在（アーカイブであることが一見わかりにくい）。
- `pyproject.toml` の `project.name` が `pyxidraw5` のまま（リポ名と不一致）。

---

## すぐ効く提案（軽微・ノンブレイキング）
- [ ] `api` の導線を `G` に一本化し、登録は `api.register_shape / api.register_effect` を推奨（すでに re-export を追加）。
- [ ] `architecture.md` に「`common` と `util` の役割境界」を 1 文追記。
      - `common`: 型/レジストリ/両翼（effects・shapes）から参照される“土台”。
      - `util`: アプリ運用・設定・小物ユーティリティ（上位からのみ参照）。
- [ ] `previous_code/` を `docs/attic/` へ移動し「過去コード置き場」と明記（履歴保持）。
      - Ask-first（破壊的移動）: 実施前に承認が必要。
- [ ] `pyproject.toml` の `project.name` を現行に合わせて更新（例: `pyxidraw6`）。
- [ ] `examples/` を新設し最小スケッチを配置（`api.runner` の導線を明確化）。

## 中期提案（任意・相談）
- [ ] `engine/monitor` を `engine/ui` 直下へ論理統合（UI 系の探索コスト低減）。
      - 影響小。ただし import パス変更が発生するため移行ガイドが必要。
- [ ] 内部専用モジュール/関数に先頭アンダースコアを付与し明示（IDE の補完ノイズ低減）。
- [ ] README/architecture.md に「層ごとの入口/出口」を1枚図で再掲（Canvas 強化）。

---

## 影響/リスク（上記の “すぐ効く提案” 実施時）
- コード移動（`previous_code/`）は Git の履歴差分が広がるため、PR を小さく保ちレビュー容易に。
- `pyproject.toml` 名称変更は配布時のメタデータに影響。社内/外部配布計画があれば合わせて確認。
- 新設 `examples/` はテスト対象外とし、README から参照するのみ（ビルドや型のノイズを避ける）。

## 完了条件（DoD）
- Lint/Type/Test 緑（編集ファイル限定の高速ループで確認）。
- ドキュメント更新済み（README / architecture.md）。
- 破壊的操作は事前承認ログを残す（この md にも記載）。

---

## 実施チェックリスト（承認後に順次着手）
- [ ] docs: `architecture.md` に `common`/`util` の役割追記
- [ ] docs: README に `G` を推奨導線として明記（`shape_registry` は内部向け）
- [ ] code: `examples/` 新設 + 最小スケッチ配置
- [ ] config: `pyproject.toml` の `project.name` を更新
- [ ] move: `previous_code/` → `docs/attic/`（Ask-first）
- [ ] docs: `architecture.md` に入口/出口の図を追記（任意）
- [ ] refactor: （任意）`engine/monitor` を `engine/ui/monitor` へ移設 + import 修正
- [ ] style: 内部専用 API に `_` 接頭辞を付与（任意・段階導入）

---

## 確認事項（ご回答ください）
1) 上記「すぐ効く提案」をこの順序で進めてよいですか？（Yes/No/修正希望）
2) `previous_code/` の移動を許可しますか？（Ask-first 対象）
3) `pyproject.toml` の `project.name` は `pyxidraw6` で問題ありませんか？
4) 中期提案のうち、優先したいものはありますか？

承認後、この md 上で進捗（チェック）を更新しながら進めます。
