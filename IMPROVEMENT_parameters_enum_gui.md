# UI 改善: エフェクトの列挙型（enum）パラメータ対応

目的
- join 等のカテゴリ変数（文字列）を GUI で直感的に操作可能にする。
- 既存の `__param_meta__`（`type: "string"`, `choices`）を一次情報源とし、最小差分で導入。

スコープ
- 対象: `effects`/`shapes` の「文字列 + choices」な引数（例: `offset.join`, `twist.axis`, `extrude.center_mode`, `text.align`, `fill.mode`）。
- 非対象: 追加の外部依存、保存形式の変更、大規模 UI リファクタ。

設計方針（最小）
- 判定: `__param_meta__` に `choices` がある、または `type == "string"` の場合に enum とみなす（優先は `choices`）。
- 表示: 候補数 2 → トグル、3〜5 → セグメントボタン、6 以上 → ドロップダウン。
- 値: 保存値は内部値（例: `'mitre'`）、表示ラベルは既定で同一文字列（将来拡張でラベル/アイコン対応可）。
- 互換: `choices` が無い文字列は従来通り GUI 非対応（passthrough のまま）。
- 不正値: `choices` に含まれない値は「未サポート値」として表示しつつ編集可能（保存時の扱いは下記質問）。

実装タスク（チェックリスト）
- [ ] `engine/ui/parameters/state.py` の `ParameterDescriptor` に `choices: list[str] | None` を追加（最小）。
  - 代替案（将来拡張）: `choices: list[str|int] | list[tuple[value,label]]` に拡張可能にしておく。
- [ ] `engine/ui/parameters/value_resolver.py`
  - [ ] `ValueType` 判定は現状の `enum` ロジックを流用（`type=="string"` or `choices` あり）。
  - [ ] `supported` を `enum` でも True にする。
  - [ ] `__param_meta__` の `choices` を `ParameterDescriptor.choices` に反映。
  - [ ] 既定値/現在値が `choices` 外の場合の扱いを決めて実装（下記質問の合意後）。
- [ ] GUI レンダラ（該当箇所）
  - [ ] `value_type=="enum"` でセレクト UI を描画。候補数に応じてトグル/セグメント/ドロップダウンを選択。
  - [ ] キーボード矢印で前後移動、数字キー（1..9）で直接選択をサポート（可能なら）。
  - [ ] MIDI マッピングは「候補インデックス（0..n-1）」にマップ（範囲外はクリップ）。
- [ ] 最小対象での確認
  - [ ] `offset.join` が GUI に表示され、選択反映される。
  - [ ] `twist.axis` が GUI に表示され、選択反映される。
  - [ ] `extrude.center_mode` が GUI に表示され、選択反映される。
  - [ ] `text.align` が GUI に表示され、選択反映される。
  - [ ] `fill.mode` が GUI に表示され、選択反映される。
- [ ] テスト（編集ファイル限定の高速ループ）
  - [ ] `ruff/black/isort/mypy` を変更ファイルに限定して通す。
  - [ ] `tests/ui/parameters` に enum の smoke テストを追加（Descriptor に `choices` が入る/`supported=True`）。
  - [ ] 可能なら `pytest -q tests/ui/parameters -k enum` 程度で分割実行。

受け入れ条件（DoD）
- 対象 5 箇所が GUI 上で選択コンポーネントとして表示され、操作がエフェクトに反映される。
- `ParameterDescriptor` に `choices` が設定され、`supported=True`。
- `ruff/black/isort/mypy`（対象ファイル）と `pytest -q -m smoke` が緑。

互換性・フォールバック
- `choices` 無しの `str` は従来通り GUI 非対応（passthrough）。
- 不正値（`choices` 外）は「未サポート値」表示。保存時の取り扱いは要合意（下記）。

将来拡張（実装外・メモ）
- `choices` のラベル化: `[{"value": "mitre", "label": "角"}, ...]` を許容。
- 並び順指定 `order`、別名 `aliases`、簡易プレビュー `icon`、依存関係 `depends_on`。
- 型注釈 `Literal[...]` からの自動抽出（`__param_meta__` に `choices` が無い場合の救済）。

質問（要確認）
1. 不正値の保存時挙動: 次のどれにしますか？
   - A) 値を維持（編集可能。変更時のみ `choices` 内へ）
   - B) 最も近い（または最初の）選択肢へ自動変換＋オーバーレイで通知
2. 表示ラベル: まずは内部値＝表示で良いか（ローカライズ不要）？
3. 並び順: `choices` の列挙順をそのまま表示順にしてよいか？
4. MIDI: インデックスマッピング（0..n-1）で問題ないか？ 範囲外はクリップで許容？

影響ファイル（予定）
- `src/engine/ui/parameters/state.py`
- `src/engine/ui/parameters/value_resolver.py`
- `src/engine/ui/parameters/`（GUI レンダ箇所）
- `tests/ui/parameters/`（新規/更新）

実行手順（ローカル）
- 初期化: `python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`
- 対象限定チェック: `ruff check --fix {changed_files}` / `black {changed_files} && isort {changed_files}` / `mypy {changed_files}`
- テスト（部分）: `pytest -q tests/ui/parameters -k enum`
- スタブ: 変更なし（公開 API には影響しない想定）

備考
- 既存の `ValueType` は `"enum"` を含んでおり、型体系上の整合は良好。
- `ParameterStore` は実値を保持しクランプしない方針なので、enum の取り扱いとも整合的。

