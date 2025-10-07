# ドキュメント整備戦略（GitHub リリース向け）

本ドキュメントは、PyXidraw を初めて知る「ジェネラティブアートに興味がある人」に最短で価値を届け、必要十分な深さで自走できるようにするための公開ドキュメント戦略。複雑化を避け、最小の導線で始めやすく、深掘りもできる構成を採用する。

## 目的と対象
- 目的: 「何ができるか」「どう始めるか」「どこを掘ればいいか」を明確化し、GitHub での初見ユーザーを素早くアクティベートする。
- 対象:
  - クリエイティブコーディング経験者（Python での生成アートに関心）
  - ラインベースの幾何とエフェクトの合成に興味（ライブコーディング/プロッタ/出力画像）
  - 初心者〜中級者（pip/venv が使える、数式/ベクトルに抵抗がない層）

## メッセージ（核となる訴求）
- ラインベースの幾何を関数で生成し、効果をパイプラインで合成、リアルタイムに描画/出力できる軽量フレームワーク。
- 形状は `G.<name>(...)`、効果は `E.pipeline.<effect>(...).build()` で宣言的に記述。
- LFO/CC を用いた時間変調やパラメータ GUI によるインタラクティブな探索が可能。
- 画像/G-code 書き出しに対応（プロッタ出力の土台あり）。

## 情報設計（Information Architecture）
最初の 60 秒で「概要→動作→発展」の順に導く。GitHub Top は README が着地点。

1) README（ランディング / 体験の入口）
- プロジェクト概要（1–2 文のタグライン + キービジュアル/GIF）
- できること（3–5 箇条の箇条書き）
- クイックスタート（環境/実行/最小例）
- さらに学ぶ（詳細ドキュメントへのリンク集）
- バッジ（テスト/ライセンス/最低 Python バージョン）

2) Getting Started（入門・最短経路）: `docs/getting-started.md`
- セットアップ（venv、pip、最低要件）
- 最小スケッチ（G/E/LFO/CC を 1 例ずつ）
- パラメータ GUI の使い方（表示条件・優先順位・RangeHint の概念）
- よくあるつまずき（DPG 未導入時の挙動、GPU なしのフォールバックなど）

3) User Guide（How-to 断片）: `docs/user-guide.md`
- 形状を作る/増やす（`src/shapes/`、`shapes/registry.py` 連携）
- 効果を合成する（`src/effects/`、`effects/registry.py`）
- LFO/CC を使う（`api.lfo`/`api.cc`、組み合わせのパターン）
- エクスポート（画像/G-code、推奨解像度/スケール）
- HUD/ランタイム（`engine/ui/hud`、`engine/runtime` の触り）

4) Examples（作例・レシピ集）: `docs/examples/README.md`
- `demo/*.py` の索引（各サムネ/出力例/ワンポイント解説）
- コードと出力の 1:1 対応（試す→見える）
- 画像は `screenshots/` に、生成は `src/engine/export/image.py` を利用予定

5) API リファレンス（軽量版）: `docs/api.md`
- 公開 API の薄い目次（`src/api/__init__.py` と `src/api/__init__.pyi` を基準）
- 代表関数のシグネチャと簡潔な説明（詳細は docstring/型へ誘導）
- スタブ生成フロー（`python -m tools.gen_g_stubs`）と CI との関係

6) アーキテクチャ（同期維持）: `architecture.md`
- すでに存在するルート `architecture.md` を正とし、README のリンク先を統一
- コア/ランタイム/レンダ/UI/IO の責務と流れ、関連テストの参照

7) 開発者ガイド（最小）: `docs/dev-setup.md`
- Build/Test/Style ルール（AGENTS.md を短く要約しリンク）
- 変更時のチェック（ruff/black/isort/mypy/pytest、スタブ更新）
- コミット規約と PR 前チェックリスト

8) 貢献ガイド/行動規範（任意）: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- 小さく始めるための導線（Issue/PR テンプレ、再現手順の書き方）

9) FAQ（任意）: `docs/faq.md`
- よくある質問（プロッタ互換、DPG 必要性、パフォーマンスの目安 等）

## 最低限の初回リリース範囲（MVP）
- README の刷新（タグライン/最小例/リンク整備/バッジ/スクショ1枚）
- Getting Started（環境/最小例/GUI/トラブルシュート）
- Examples 索引（最低 3 本を選抜、`demo/` 連携）
- Architecture のリンク整合（README → `architecture.md` を正とする）

## Nice to have（M1）
- User Guide（How-to 分割）
- API リファレンス（主要 API の薄いまとめ）
- Developer Guide（AGENTS.md の要約版）

## あると嬉しい（M2）
- FAQ、CONTRIBUTING、テンプレ群、英語 README の簡易版（翻訳）
- スクショ/GIF 自動生成スクリプト（将来）

## 作業計画（チェックリスト）
- README 再編（リンク/断言の整合を確保） [ ]
  - `docs/architecture.md` → 実体は `architecture.md` のためリンク修正 [ ]
  - `docs/dev-setup.md` は未作成 → 最初は README に簡易版を直書き、追って分離 [ ]
  - `docs/effects.md` は未作成 → 初回は README から削除/名称調整、追って作成 [ ]
  - スクリーンショット 1 枚（`screenshots/hero.png`）の用意 [ ]
- Getting Started 新規: `docs/getting-started.md` [ ]
  - セットアップ/最小スケッチ/GUI/トラブルシュート [ ]
- Examples 索引: `docs/examples/README.md` [ ]
  - `demo/shape_grid.py`/`demo/effect_grid.py`/`demo/define_your_shape.py` を採用 [ ]
  - 画像は一旦プレースホルダ（後で `engine/export/image.py` で生成） [ ]
- API 概要: `docs/api.md`（軽量、`api/__init__.pyi` 準拠） [ ]
  - スタブ生成手順の記載（`python -m tools.gen_g_stubs`） [ ]
- Developer Guide: `docs/dev-setup.md`（AGENTS 要約） [ ]
  - Build/Test/Style/Stub/CI を 1 画面で俯瞰 [ ]
- README の英語サマリ（任意、末尾に短段落） [ ]
- バッジ追加（pytest/ライセンス/Python） [ ]
- Issue/PR テンプレ（任意） [ ]

## コンテンツ方針（スタイル）
- 短く・具体的・反復可（AGENTS 準拠）。図は最小限、コードで語る。
- 例は実行可能で、`demo/` と 1:1 に対応させる。
- 「Why/トレードオフ」は最小限（詳細はコードコメントや ADR に譲る）。
- 公開 API は「薄い目次 + 型/スタブへの誘導」。重い Doc ツールは導入しない。

## リンク/用語の整合性
- 「アーキテクチャ」は `architecture.md` を正とする（README のリンク修正）
- 「パイプライン/API」は `docs/pipeline.md` を参照
- 「LFO 仕様」は `docs/lfo_spec.md` を参照
- 形状/効果の一覧は将来 `docs/shapes.md`/`docs/effects.md` に分離（当面は README の最小リンクのみ）

## スクリーンショット運用（軽量）
- フォルダ: `screenshots/`（PNG）
- 生成方針: `src/engine/export/image.py` を使った固定シードの再現可能生成
- 命名: `hero.png`, `example_shape_grid.png`, `example_effect_grid.png`

## SEO/発見性
- GitHub Topics: `generative-art`, `creative-coding`, `python`, `vector-graphics`, `plotter`, `axidraw`, `dearpygui`
- 冒頭タグラインで「ラインベース」「リアルタイム」「パイプライン」を明記

## リリース手順（ドキュメント観点）
1) チェックリスト Mvp をすべて埋める
2) `ruff/black/isort/mypy/pytest -q -m smoke` を変更ファイルに対して通す
3) README のリンク切れ確認（ローカル相対パス/存在性）
4) スクショがある場合は容量確認（軽量）
5) CI バッジ/スタブ生成の案内が最新であること

## 既知の差分/課題（要確認）
- README が `docs/architecture.md` を参照しているが、実体はルート `architecture.md`（リンク修正が必要）
- `docs/dev-setup.md`, `docs/effects.md` は未作成（初回は README からリンクを外す or プレースホルダを置く）

---

この戦略案で進めて問題なければ、MVP 範囲の 4 点（README 再編 / Getting Started / Examples 索引 / Architecture リンク整合）から着手し、PR を段階投入します。改善中の気づきや要相談事項は本ファイルに追記し、チェックリストで進捗を可視化します。
