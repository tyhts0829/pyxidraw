# Geometry 正規化リライト計画（improvement_tasks_20250916 #3）

## フィロソフィ
- 破壊的変更も許容し、`Geometry` を「常に美しく正規化された状態だけが存在する」クラスへ作り替える。
- 内部仕様を 1 箇所に集約して重複をなくし、読む側が一度で理解できるシンプルなコードを目指す。

## 方針
- dataclass 依存をやめ、`__slots__` を持つ明示的な `__init__` に置き換え、生成フローを 1 本化する。
- `Geometry` 生成時は必ず `_normalize_arrays(coords, offsets)` のような私的ヘルパで整形。`from_lines` も同ヘルパへ委譲し二重処理を排除。
- 入力が破壊的に変換されることを許容し、外部 API にも dtype/shape の厳格化を強制する。

## 対象範囲
- `src/engine/core/geometry.py`: `Geometry` クラス全体をリライト（`__init__`, `from_lines`, 付随メソッドの微調整）。
- `tests/core/`: dtype・shape 正規化、例外ケース、API 破壊変更に伴う更新。
- `architecture.md` / `Geometry` docstring: 新しい前提へ更新。

## 実装タスクリスト
- [ ] 現行 `Geometry` 利用箇所（特に `Geometry(...)` 直接呼び出し）を洗い出し、新しい厳格仕様に合わせて書き換えの影響を把握する。
- [ ] `Geometry` を dataclass から通常クラスへ置換し、`__slots__ = ("coords", "offsets", "_digest")` を定義。
- [ ] `_normalize_geometry_input(coords, offsets)` を実装。
  - `coords`: `np.asarray(coords, dtype=np.float32)` → `ndim==2`, `shape[1]==3`, C-contiguous を保証。
  - `offsets`: `np.asarray(offsets, dtype=np.int32)` → `ndim==1`, 先頭0/末尾`len(coords)`/単調非減少を検証。
  - 条件違反時は即 `ValueError`。必要最小限の `astype(copy=False)`/`np.ascontiguousarray` でコピー回数を抑制。
- [ ] `__init__` で上記ヘルパの戻り値を受け、内部状態を直接設定。`from_lines` は入力構築後に `cls(coords, offsets)` を呼ぶだけにする。
- [ ] `translate/scale/rotate/concat/as_arrays` 等の内部実装を新仕様に合わせて調整（戻り値も厳格化維持）。
- [ ] 破壊的変更に合わせ tests を更新/追加。
  - float64/int64/不正形状 → `ValueError`
  - 正常入力は float32/int32 に落ちること、C-contiguous であること
  - `concat` 等の結果 dtype を確認
- [ ] `architecture.md` と docstring を新しい「コンストラクタが正規化を担保する」記述へ同期。
- [ ] 完了後に実行する検証コマンドを決定（`ruff`/`mypy`/`pytest tests/core/test_geometry*.py`）。

## 未決事項・検討
- offsets の平坦化：ゼロ長セグメント（`offsets[i]==offsets[i+1]`）を許容するか要確認。単調非減少の範囲で現仕様を踏襲する想定。
- `concat` の内部で `np.vstack`/`np.hstack` する際も最終的に `_normalize_geometry_input` を通すか、既に規約を満たす配列を直接利用するか（パフォーマンス vs 一貫性）。
- API 破壊の周知: 外部で float64 を期待している既存コードへの影響がないか確認（配布前リポのため重大ではない想定）。
