## 基本クリーニング指摘とチェックリスト

- 作成日: 2025-09-08
- 対象コミット: e1f5813（ローカル現在）
- 目的: リポジトリ内の「基本的なクリーニング」が未実施/不統一な箇所を可視化し、実施順に解消できるようチェックリスト化する。

### メタ（本タスク）

- [x] リポジトリ走査（構成/設定/CI/追跡ファイルの確認）
- [x] 本チェックリストの作成と保存（REPO_CLEANUP_CHECKLIST.md）

---

### 1) 依存関係・パッケージング

- [x] ランタイム依存 `numpy` を `[project].dependencies` に追加（現状は空）。
  - 事実: ランタイムで `numpy` を多数インポート（例: `src/engine/core/geometry.py`, `src/effects/*`, `src/shapes/*`, `src/api/*`）。
  - 影響: `pip install .`（dev/optional 以外）で実行時に ImportError のリスク。
  - 対応例: `pyproject.toml` の `[project] dependencies = ["numpy>=1.23"]`（必要最小に合わせて調整）。
- [ ] `dev`/`optional` の重複依存（例: `fonttools`, `fontpens`）の整理（どちらに寄せるか方針化）。
- [ ] `data/` 配下の生成スクリプト（例: `data/sphere/sphere.py`）はランタイム依存を不要化 or `scripts/` 配下へ移動（役割の明確化）。

### 2) CI / 自動化の不整合

- [x] スタブ更新チェックの対象パスを修正。
  - 現状: `.github/workflows/verify-stubs.yml` が `api/__init__.pyi` を比較。
  - 事実: 生成物は `src/api/__init__.pyi`（`src/scripts/gen_g_stubs.py` 参照）。
  - 対応例: `git diff --exit-code src/api/__init__.pyi` に修正。
- [x] Python バージョン表記の統一（CI は 3.10 固定）。
  - ドキュメントでは 3.11 推奨の記述が混在（詳細は「ドキュメントの不統一」を参照）。

### 3) Git 追跡状態・不要/生成物の扱い

- [x] 生成物/デバイス固有ファイルが Git 追跡下にあるため除外・移動。
  - 実施: `src/engine/io/cc/*.pkl` / `*.json` を `data/cc/` へ移動（ファイル名はユーザ指定により変更せず）。
  - 参照更新: `engine/io/controller.py` の保存先を `data/cc/` に切替（`PXD_DATA_DIR` で上書き可）。
  - 一時対応: 履歴保持不要なら `git rm -r --cached src/engine/io/cc`（必要時）。
- [ ] `.pydeps` が追跡中（用途明確化のうえ、不要なら除外/削除）。
- [ ] 古い VS Code ワークスペース名の残存。
  - 対象: `pyxidraw4.code-workspace`（プロジェクト名と不一致）。
  - 対応: リネーム（例: `pyxidraw.code-workspace`）または削除。
- [ ] `screenshots/` を .gitignore から外す（PR での見た目差分要求と矛盾）。
  - 現状: `.gitignore` に `screenshots/` が記載。
  - 方針: プレビュー画像は追跡対象にする（大容量は最小限・圧縮）。

### 4) ドキュメントの不統一/リンク切れ

- [x] Python バージョン表記の統一。
  - 事実: `AGENTS.md` は「Python 3.10 推奨」、`docs/dev-setup.md`/`README.md` は 3.11 を使用例に記載。
  - 方針: CI と整合する 3.10 を推奨に統一、使用例も `python3.10 -m venv` に合わせる。
- [x] README のリンク切れ修正。
  - 対象: `TEST_PLAN.md`, `TEST_HARDENING_PLAN.md` へのリンクが存在するがファイル未検出。
  - 対応: ファイルを追加するか README から項目を削除/置換。
- [ ] 名称の統一。
  - 事実: プロジェクト名の表記が `PyXidraw5`（README/pyproject）と作業ディレクトリ名（`pyxidraw6`）/ワークスペース名（`pyxidraw4`）で混在。
  - 方針: 公称名（例: `PyXidraw5`）に統一、ワークスペース/フォルダ名も合わせる。

### 5) コードスタイル/言語ポリシー（日本語化・命名）

- [x] docstring の日本語統一（主要モジュールを一括対応）。
  - 例: `src/shapes/attractor.py` クラス/メソッドの docstring、`src/effects/*`、`src/engine/core/*` など。
  - 推奨書式:
    - セクション見出しを「引数:」「返り値:」「例:」に統一。
    - 用語の日本語/英語の併記ルールを決める（初出で英語をカッコ書き）。
- [ ] ファイル名の空白を廃止し `snake_case` に統一。
  - 例: `src/engine/io/cc/*TX-6 Bluetooth*.pkl` など。
- [x] `data/` 配下のスクリプト整備。
  - 例: `generate_shpere_*` のスペル修正（`sphere`）、型ヒント追加、ワンショット実行手順を README に追記。

### 6) .gitignore の見直し

- [x] 重複/冗長パターンの整理。
  - 現状: `engine/io/cc/` と `src/engine/io/cc/` の重複。
  - 方針: `src/engine/io/cc/` のみで十分（`src/` レイアウトに合わせる）。
- [x] 追跡したいアセット（`screenshots/`）は除外しない方針に変更。

---

### 付録: 代表的な修正コマンド例

- スタブチェックのパス修正（CI）
  - `.github/workflows/verify-stubs.yml` 内を `api/__init__.pyi` → `src/api/__init__.pyi` に置換。
- 追跡済み生成物の除外（履歴温存不要の簡易版）
  - `git rm -r --cached src/engine/io/cc && git add -A && git commit -m "chore(git): untrack generated cc artifacts"`
- `screenshots/` の追跡化
  - `.gitignore` から該当行を削除し、必要に応じて `screenshots/` に `.gitkeep` を配置。
- 依存関係の最小化/明確化
  - `pyproject.toml` の `[project].dependencies` に `numpy` を追加し、`dev` に重複があれば整理。

---

### 実施順の提案（優先度高 → 低）

1. 依存関係（`numpy` 追加）と CI のスタブチェック修正。
2. 生成物の追跡解除/移動（`src/engine/io/cc`）。
3. Python バージョン表記と README のリンク修正。
4. `.gitignore` 整理と `screenshots/` の追跡化。
5. docstring の日本語統一とファイル名の空白除去（段階的に）。

以上。各項目の対応が完了したら本ファイルのチェックを更新してください。

---

## 要確認事項（承認待ち）

以下の各項目について、該当する選択肢に [x] を入れてください。必要に応じて「備考」に自由記入してください。

1. `src/engine/io/cc` の生成物（`*.pkl`/`*.json`）の扱い

- [ ] A. 履歴からの追跡解除のみ（削除、.gitignore 維持）
- [x] B. `data/` へ移動し参照コード/テストを更新
- [ ] C. 現状維持（理由を備考に記載）
- ファイル名の空白を `snake_case` に改名してよいか: [ ] OK / [x] NG
- 備考:デバイス固有の名前に空白が入っているのでどうようもない
  実施: `data/cc/` へ移動完了。`MidiController` の保存先も `data/cc/` へ変更。

5. `data/sphere/sphere.py` の整理

- [x] `scripts/` へ移動（データ生成ツールとして扱う）
- [ ] 現状維持 / [ ] 削除
- 関数名スペル修正（`generate_shpere_*` → `generate_sphere_*`）: [ ] OK / [ ] NG
- `trimesh` を `optional` 依存へ追加: [ ] OK / [ ] NG
- 備考:
  実施: `src/scripts/sphere.py` へ移動（関数名は現状維持）。

7. docstring の日本語化ポリシー

- [x] 一括対応（全体）
- [ ] 段階対応（変更頻度の低いモジュールから）
- pre-commit で軽いチェック（`Args/Returns` → `引数/返り値`）を追加: [ ] OK / [ ] NG
- 備考:
  実施: 主要モジュール（`engine/core/geometry.py`, `engine/core/render_window.py`,
  `engine/io/helpers.py`, `shapes/attractor.py`, `shapes/base.py`, `effects/{scale,explode,trim,twist,displace,weave}.py` など）
  を日本語表記へ統一。残タスクは順次対応。

8. そのほか命名/配置の微修正

- `src/engine/io/cc` 以外で空白を含むファイル名の一括是正: [ ] OK / [x] NG
- `data/` 配下のスクリプトを `scripts/` に集約: [x] OK / [ ] NG
- 備考:
