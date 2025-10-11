# フォント探索を `configs/default.yaml` のフォルダから行う改善計画

目的: `src/shapes/text.py`（Text シェイプ）と UI（HUD/Overlay、必要なら Parameter GUI）で使用するフォントを、OS 既定のパス探索に加え、`configs/default.yaml`（→ `config.yaml` で上書き可）で指定したフォルダから優先的に探索・利用できるようにする。

---

## 背景 / 現状
- Text シェイプ（`src/shapes/text.py`）は OS 依存ディレクトリ（macOS/Linux/Windows）を再帰探索してフォントファイル（`.ttf/.otf/.ttc`）を列挙し、部分一致またはパス直指定で選択する実装。
- HUD オーバーレイ（`src/engine/ui/hud/overlay.py`）は `pyglet.text.Label` に固定文字列の `font_name`（例: `HackGenConsoleNF-Regular`）を渡している。設定からのフォント名/サイズの読み込みや、任意ディレクトリからのフォント登録は未対応。
- 設定読み込みは `util.utils.load_config()` が提供（`configs/default.yaml` → ルート `config.yaml` の順に上書き）。現状 `fonts` セクションは未定義。

## 目標
- 設定 `fonts.search_dirs` で指定したフォルダ（複数可）を再帰的に走査し、
  - Text シェイプのフォント探索リストに「最優先」追加する。
  - HUD で `pyglet` にフォントを登録して `font_name` 解決に使えるようにする。
- HUD のフォント名/サイズを設定から読み取れるようにする（`hud.font_name`, `hud.font_size` を優先し、互換として `status_manager.font`/`status_manager.font_size` も読む）。
- 既存の引数/API は変更しない（後方互換）。設定未指定時は従来挙動を維持。

## 非目標
- フォントの埋め込み/配布や、フォントライセンス対応の自動化は行わない。
- Dear PyGui 側のカスタムフォントバインドは第2段階（任意）。本計画では HUD のみ確実に対応。

---

## 設計方針
- 設定スキーマ（追加）
  - `fonts.search_dirs: ["data/fonts", "~/Library/Fonts/Custom", ...]`
    - 文字列 or 配列を許容。環境変数と `~` 展開を行う。
    - 相対パスは「プロジェクトルート基準（`util.utils._find_project_root`）」で解決。
  - `hud.font_name: "HackGenConsoleNF-Regular"`（省略可）
  - `hud.font_size: 12`（省略可）
  - 後方互換: `status_manager.font` / `status_manager.font_size` を HUD の既定として参照。
- 優先順位（探索）
  1) `fonts.search_dirs`（再帰, `*.ttf|*.otf|*.ttc`）
  2) OS 既定ディレクトリ（現状実装）
  3) OS 別の既定候補（現状実装）
- キャッシュ
  - `TextRenderer._font_paths` を踏襲。初回計算時に設定ディレクトリを前置して構築。
  - 実装簡素化のためホットリロードは対象外（再起動で反映）。
- HUD（pyglet）
  - 設定ディレクトリ配下の `*.ttf/*.otf/*.ttc` を列挙し、`pyglet.font.add_file(path)` で登録。
  - `.ttc` のサポート状況に応じて失敗は握りつぶして継続（ベストエフォート）。
  - フォント名/サイズは設定から読み取り、未指定時は既定値を維持。

---

## 実装タスク（チェックリスト）
- [ ] 設定スキーマの追加: `configs/default.yaml` に `fonts.search_dirs` と HUD 用 `hud.font_name`/`hud.font_size` を追記（コメントで意味と例を併記）。
- [ ] Text シェイプの探索強化: `src/shapes/text.py`
  - [ ] `get_font_path_list()` を拡張し、`load_config()` から `fonts.search_dirs` を読み込む。
  - [ ] 検出ディレクトリを（OS 既定より）前に結合。`~`/env/相対パス解決と再帰グロブを実装。
  - [ ] 既存の拡張子フィルタ（`.ttf/.otf/.ttc`）を再利用。
  - [ ] 例外時は安全にスキップしてOS探索へフォールバック。
- [ ] HUD 側のフォント登録: `src/engine/ui/hud/overlay.py`
  - [ ] 初期化時に `fonts.search_dirs` を読み取り、該当パス配下のフォントを `pyglet.font.add_file()` で登録（失敗は握りつぶし）。
  - [ ] `hud.font_name`/`hud.font_size`（→ 無ければ `status_manager.*` → 既定）で `self._font`/`self.font_size` を上書き。
  - [ ] 既存の色/メータ設定ロジックと併存させる。
- [ ] ドキュメント整備
  - [ ] `architecture.md` に「フォント探索の優先順位」「設定キー」「HUD 登録の流れ」を追記し、該当コード参照（`src/shapes/text.py`, `src/engine/ui/hud/overlay.py`）を明記。
- [ ] 最小テスト
  - [ ] 単体: Text シェイプの `get_font_path_list()` が設定ディレクトリを優先に含むことを検証（ダミー拡張子ファイル名の列挙レベルで可。実フォント読み込みは不要）。
  - [ ] 既存テストのスモーク: `pytest -q -m smoke` で影響の有無を確認。
  - [ ] 変更ファイルに限定して `ruff/mypy` を実行。

---

## 変更ファイル（想定）
- `configs/default.yaml`（設定キー追加）
- `src/shapes/text.py`（探索リスト構築の拡張）
- `src/engine/ui/hud/overlay.py`（フォント登録と設定反映）
- `architecture.md`（設計同期）
- （任意）`tests/` に最小の追加テスト

---

## 受け入れ条件（DoD）
- 設定未指定時は従来どおり OS フォント探索で動作する。
- `config.yaml` に `fonts.search_dirs` を指定すると、その中に置いたフォント名（例: `MyFont-Regular`）を Text シェイプの `font` 引数や HUD の `font_name` で解決できる。
- 変更ファイルに対する `ruff/mypy/pytest -q -m smoke` が緑。
- `architecture.md` に実装との齟齬がない。

---

## リスクと対応
- 大量のフォントを含むディレクトリを指定した場合の初回探索コスト → 拡張子フィルタの厳格化と結果のキャッシュで軽減。
- `.ttc` の `pyglet.font.add_file()` 対応差 → 失敗を握りつぶして続行（他の拡張子で補う）。
- 相対パス/ホーム展開の齟齬 → プロジェクトルート基準で統一し、`~`/環境変数を展開。

---

## 追加の確認事項（ご承認お願いします）
1) 設定キー名: `fonts.search_dirs` / `hud.font_name` / `hud.font_size` で問題ありませんか？（互換参照: `status_manager.*`）
2) `fonts.search_dirs` は再帰探索で問題ありませんか？（直下のみ希望なら切替可能）
3) HUD の既定フォントは現状どおり `HackGenConsoleNF-Regular` を維持しつつ、設定で上書きする方針でよいですか？
4) Dear PyGui についても、同様のフォント登録/適用を次段で対応しますか？（任意）

---

## 作業/検証コマンド（変更ファイル優先）
- Lint: `ruff check --fix {changed_files}`
- Format: `black {changed_files} && isort {changed_files}`
- TypeCheck: `mypy {changed_files}`
- Test (smoke): `pytest -q -m smoke`
- 追加テスト（例）: `pytest -q tests/shapes/test_text_font_search.py::test_config_dirs_prepend`

---

以上の計画でよろしければ、実装に着手します。修正途中の論点や改善提案は本ファイルに追記して進捗共有します。

