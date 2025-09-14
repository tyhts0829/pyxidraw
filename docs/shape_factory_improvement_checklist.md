# ShapeFactory 改善アクション チェックリスト

対象: `src/api/shape_factory.py`
目的: 設計ドキュメントとの整合・DX 向上（動的メソッドのメタデータ付与）・必要なら旧形式 `(coords, offsets)` 互換の整理。

このチェックリストは AGENTS.md の運用規約に基づく。実装前に要承認項目の方針を決定してください。

---

## 要承認（選択が必要）

- [x] 旧形式 `(coords, offsets)` の扱いを決定する（どちらか一方を選択）
  - [ ] A: 実装で互換サポートを追加（`_cached_shape` で明示分岐・バリデーション）
  - [x] B: 旧形式はサポート対象外とし、モジュール先頭 Doc と `architecture.md` を現状に整合（記述を削除/訂正）
- [x] 動的メソッド（`G.<name>`）のメモ化を行うか
  - [x] する（ガード付きメモ化。`setattr` でクラスに `staticmethod` を一度だけバインド）
  - [ ] しない（現状維持。レジストリ動的変更の即時反映を優先）

---

## 実装タスク（方針決定後に着手）

1. 旧形式 `(coords, offsets)` 互換（A を選んだ場合のみ）

- [ ] 受理条件の実装: `tuple` 長さ 2、両要素が `np.ndarray`、dtype/shape を検証
- [ ] `Geometry(coords.astype(float32), offsets.astype(int32))` で組み立て
- [ ] 異常系の明確化: 形状不一致や dtype 不正時は `ValueError`
- [ ] テスト追加: `tests/api/test_shape_factory_legacy.py::test_accepts_coords_offsets_tuple`

2. ドキュメント整合（B を選んだ場合のみ）

- [x] `src/api/shape_factory.py` 冒頭 Doc の旧形式言及を削除/訂正（旧タプル形式は非サポートを明記）
- [x] `architecture.md` の関連箇所を検索し、該当記述なしを確認（更新不要）

3. 動的メソッドのメタデータ付与（共通）

- [x] `__name__`, `__qualname__`, `__doc__` を設定（`shape_cls.generate` から doc を流用）
- [ ] `functools.wraps` の適用（可能な範囲）
- [ ] テスト追加（任意）: `G.sphere.__doc__` に一部文字列が含まれることを確認

4. 動的メソッドのメモ化（選択した場合）

- [x] クラス側 `ShapeFactoryMeta.__getattr__` で `setattr(cls, name, staticmethod(_shape_method_impl))`
- [x] インスタンス側 `ShapeFactory.__getattr__` はクラス属性に委譲（メモ化を共有）
- [x] 注意点を実装コメントに追記（呼び出し時ガードで未登録化を検知し、属性削除＋AttributeError）

5. 型/スタイル整備（小修正）

- [x] `ParamsTuple` の `object`→`Any` への統一（コード規約に合わせる）
- [x] 行長 100 の超過箇所を調整（必要最小限）

6. 検証（編集ファイル優先の高速ループ）

- [x] Lint: `ruff check --fix src/api/shape_factory.py`
- [x] Format: `black src/api/shape_factory.py && isort src/api/shape_factory.py`
- [x] Type: `mypy src/api/shape_factory.py`（変更ファイルに限定。テスト側の型は段階導入のため未実施）
- [x] Test: `pytest -q tests/api/test_shape_factory.py`（既存テスト緑）
- [x] スタブ同期が不要であることを確認（`scripts/gen_g_stubs.py` の前提に影響なし）

7. ドキュメント更新（必要時）

- [x] `src/api/shape_factory.py` のモジュールドキュメント整形（NumPy スタイル節の調整）
- [x] `architecture.md` の差分を最小で同期（参照コード箇所の明記）

---

## 影響範囲とリスク

- 互換分岐を追加する場合、入力解釈の曖昧さが入るため厳格なバリデーションを行う。
- メモ化を有効化すると、実行中にシェイプが追加/解除された場合の反映が遅延する（プロセス再起動 or 明示的な `delattr` が必要）。
- いずれも `Geometry` の API には変更を与えない。

## ロールバック指針

- 互換分岐は単一関数内のガードであり、削除が容易。
- メモ化は `setattr` の追加のみで、削除で現状復帰可能。

## 完了条件（DoD）

- 変更ファイルに対する `ruff/black/isort/mypy` が成功。
- 既存 `tests/api/test_shape_factory.py` と追加テストが緑。
- 必要に応じてドキュメント（`architecture.md`/モジュール Doc）が整合。

---

## メモ / 質問事項（追記欄）

- 互換性方針: 過去のユーザコードで `(coords, offsets)` タプルを直接返していた実装が実在するか？（存在しなければ B 推奨）
- ランタイムでのシェイプ登録/解除をユースケースとして想定しているか？（想定するならメモ化は保留、しないなら DX 目的で有効化）

---

作成者: Codex CLI エージェント

---

## 実施ログ（抜粋）

- 実装: `src/api/shape_factory.py` にガード付きメモ化を追加、関数メタデータを付与。
- Lint/Format: `ruff check --fix`, `black`, `isort`（変更ファイル）
- Type: `mypy src/api/shape_factory.py`（Success: no issues found）
- Test: `pytest -q tests/api/test_shape_factory.py`（5 passed）
- Docs: モジュール Doc を `Notes/Design/Examples` に整理し、LRU/並列安全メモを `Notes` に統合。
- Docs: `architecture.md` に旧形式非サポートの参照（`src/api/shape_factory.py`）を明記。
