# grid.subdivisions → nx/ny(int) への移行計画（提案）

目的

- 線数（分割数）のような“整数カウント”を意味するパラメータを、意味に沿った int で扱う。
- Parameter GUI でも int スライダーとして自然に操作できるようにし、意味の明確さと一貫性を高める。

背景

- 現状 `grid(subdivisions=(nx, ny))` の Vec2（実数）を受け取り shape 側で `int(round(...))` による丸めをしている。
- 意味的には整数であるため、API/GUI/保存で int として扱う方が明確かつシンプル。
- 本リポの方針（明確さ・シンプル優先、破壊的変更可）にも適合。

方針（ベストプラクティス）

- `grid(nx: int = 20, ny: int = 20)` の 2 つの独立パラメータに分解する。
- `subdivisions` は廃止（互換レイヤは設けない）。
- 既存スクリプトや保存済み GUI オーバーライドの互換調整は行わない（エラー許容）。

影響範囲

- 形状 API: `src/shapes/grid.py`（シグネチャ/param_meta の変更）
- 公開スタブ: `src/api/__init__.pyi`（`G.grid` のシグネチャ）
- 参照コード: `tests/perf/*`, `docs/spec/shapes.md`, `sketch/showcase/*`, 他 `G.grid(subdivisions=...)` を使う箇所
- Parameter GUI: int スライダーで `nx/ny` を表示（既存機能で対応可）

仕様（詳細）

- シグネチャ: `def grid(nx: int = 20, ny: int = 20, **params: Any) -> Geometry`
- 範囲/ヒント: `1 <= nx, ny <= 100`（実クランプはしない、GUI レンジのみ）
- param_meta（GUI ヒント）:
  - `{"nx": {"type": "integer", "min": 1, "max": 100, "step": 1}, "ny": {...同}}`
- 実行ロジック: 変更なし（`_generate_grid(nx, ny)`）。
- キャッシュ署名: int は量子化不要（現行ルール踏襲）。

互換/移行

- 互換変換（`subdivisions` → `nx/ny`）は実施しない。
- `G.grid(subdivisions=(a,b))` を使用するコードは `G.grid(nx=a, ny=b)` に修正が必要。
- 既存 GUI 保存値（例: `shape.grid#0.subdivisions`）は無視される or 参照時にエラー。ユーザー操作で削除/更新してもらう。

タスク一覧（チェックリスト）

- [x] 仕様確定
  - [x] `nx/ny` の int 化（本書方針）
  - [x] 互換レイヤ無し（エラー許容）
  - [x] `MAX_DIVISIONS` の廃止（param_meta.max を 100 に設定）

- [x] 実装（shape）
  - [x] `src/shapes/grid.py` のシグネチャを `nx:int, ny:int` に変更
  - [x] `__param_meta__` を `nx/ny` の integer に更新（`subdivisions` 削除）
  - [x] 丸め/クランプを int 前提に簡素化

- [x] スタブ更新
  - [x] `PYTHONPATH=src python -m tools.gen_g_stubs` で `src/api/__init__.pyi` を再生成
  - [ ] `tests/stubs/test_g_stub_sync.py` を緑化

- [x] リファレンス/用例更新
  - [x] `docs/spec/shapes.md` の記述を `grid(nx, ny)` に変更
  - [x] `tests/perf/test_catalog_perf.py`/`tests/perf/test_pipeline_perf.py` を置換
  - [ ] `sketch/showcase/shape_grid.py` / `sketch/showcase/effect_grid.py` の呼び出しを更新

- [ ] 検証
  - [x] 変更ファイルに対する ruff/mypy を通す（対象限定）
  - [ ] 関連テストを最小で更新し pytest を実行（任意）
  - [ ] Parameter GUI 上で `nx/ny` が int スライダーで表示・編集できることを手動確認

- [ ] ドキュメント/整合
  - [ ] `AGENTS.md` と `architecture.md` の差分チェック（必要なら更新）

ノート/リスク

- ベクトル型 UI を介さないため、先行の Vec2 導入とは独立に安全。
- 破壊的変更のため、`subdivisions` 参照箇所の置換漏れに注意（rg で網羅）。

参考（変更前 → 変更後の例）

- 変更前: `G.grid(subdivisions=(100, 120)).scale(200, 200, 1)`
- 変更後: `G.grid(nx=100, ny=120).scale(200, 200, 1)`

確認事項（要回答）

1. `subdivisions` は完全廃止でよいか（互換エイリアスなし）。はい
2. `MAX_DIVISIONS=100` を据え置きで良いか（増減の希望があれば指定）。これも廃止。param meta で max を 100 にしておいて。
3. テスト・用例の一括置換は私の側で実施して問題ないか。　はい
