# src/effects + src/shapes トップレベル try-import / sentinel 違反チェックリスト

対象範囲: `src/effects/`, `src/shapes/`

適用ルール（本項のみ）
- トップレベルの try-import/sentinel/専用例外は用いない（ImportError はそのまま上げる）。

---

## 検出結果（違反箇所の列挙: ファイル:行）

トップレベルの try-import（オプショナル依存フォールバックを含む）
- src/effects/dash.py:30
- src/effects/collapse.py:149
- src/shapes/text.py:20

トップレベルの sentinel（依存有無や状態を示すフラグ）
- src/effects/collapse.py:152（NUMBA_AVAILABLE = True）
- src/effects/collapse.py:154（NUMBA_AVAILABLE = False）
- src/effects/partition.py:33（_HAS_SHAPELY = None）

備考
- 関数内の try-import は本ルールの対象外（トップレベル禁止のみ）。

---

## 修正アクション（本ルールに限定した最小チェックリスト）

- [ ] A. トップレベル try-import を撤去して ImportError をそのまま上げる
  - [ ] src/effects/dash.py:30 の `try: from numba import njit ... except ...` を削除
  - [ ] src/effects/collapse.py:149 の `try: from numba import njit ... except ...` を削除
  - [ ] src/shapes/text.py:20 の `try: from numba import njit ... except ...` を削除

- [ ] B. トップレベル sentinel を撤去
  - [ ] src/effects/collapse.py:152,154 の `NUMBA_AVAILABLE` 定義を削除
  - [ ] src/effects/partition.py:33 の `_HAS_SHAPELY = None` を削除

---

## メモ（実装時の注意）

- 削除後に必要な分岐やフォールバックを維持する場合でも、トップレベルではなく関数内で扱うこと（ただし本チェックリストのスコープ外）。
- 本変更は ImportError を起動時ではなく呼び出し時に発生させる可能性があり、ユースケースによってはテストの調整が必要です。

