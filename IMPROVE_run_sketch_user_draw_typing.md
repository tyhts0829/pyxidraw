# 改善提案: run_sketch の user_draw 型注釈整合性

目的: src/api/sketch.py の `run_sketch(user_draw, ...)` における `user_draw` の型注釈が実装実態より狭く、Pylance で警告が出る問題を解消する。

現状の問題
- 期待型: `run_sketch` は `user_draw: Callable[[float], Geometry]` を要求。
- 実装実態: ワーカー側は `LazyGeometry` や `Sequence[Geometry|LazyGeometry]` を許容し、レイヤー正規化して処理。
- 結果: `Geometry | LazyGeometry` を返す `user_draw` を渡すと Pylance が不一致として警告。

原因
- 型注釈の不整合（API宣言が狭い）。呼び出し側の一般的用法（`G`/`E` により Lazy を返す）が注釈と矛盾。

改善アクション（チェックリスト）
- [x] 1) `src/api/sketch.py` の `run_sketch` 引数 `user_draw` を実体に合わせて広げる
  - 目標型: `Callable[[float], Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]]`
  - Docstring も同様に「Geometry または LazyGeometry（およびレイヤー配列）」を許容と明記
- [x] 2) `src/engine/runtime/worker.py` の `_WorkerProcess.__init__` と `WorkerPool.__init__` の `draw_callback` 注釈を同じ型に拡張
- [x] 3) 参照先の共通化: `_execute_draw_to_packet` のシグネチャは既に広いので変更不要であることを確認のうえコメントを添える
- [x] 4) `src/engine/runtime/packet.py` は既に `Geometry | LazyGeometry | None` のため変更不要であることを確認
- [x] 5) 公開 API ドキュメント（`src/api/__init__.py` の Usage 説明・`run_sketch` 説明）との整合を軽く点検し、必要なら注釈/説明の最小修正

任意改善（要確認）
- [ ] A) `run_sketch` docstring の引数説明中で「CC は cc[i]」の記載を保持しつつ、戻り値例に Lazy やレイヤー化（スタイル付き）を追加
- [ ] B) `architecture.md` に「user_draw の戻り値」が Geometry 限定で記載されている場合のみ、実装差分に合わせて文言更新（該当箇所の行参照を添付）

非目標（今回含めない）
- 既存レンダラや HUD、パラメータ GUI のロジック変更
- 実行時の挙動変更（型注釈と説明の整合まで）

検証（編集ファイル限定で実行）
- [x] `ruff check --fix {changed}` / `black {changed} && isort {changed}`
- [x] `mypy {changed}`
- [ ] `pytest -q -m smoke` もしくは、`tests` に該当があれば対象テストのみ（未実施）

影響範囲とリスク
- 公開シグネチャを広げるのみ。後方互換性は維持。
- 既存利用コードはそのまま（狭/広い戻り値の双方を許容）。

確認したい点（ご回答ください）
1. `user_draw` はレイヤー（配列）も正式にサポートして良いか（実装は既に対応済み）
2. `architecture.md` の更新可否（該当記述があれば最小修正）
3. Docstring のサンプルに Lazy を返す短例を1つ追加して良いか

合意後の進め方
- 合意いただければ 1)～5) を実装 → 変更ファイル限定の型/整形/最小テストを実行 → 本 MD にチェックを付与して完了報告します。
