# effects/shapes における numba 用環境変数の使用状況

目的: effects および shapes モジュールのうち、numba を利用し、かつ numba の有効/無効や挙動を環境変数で切り替える実装がある箇所を一覧化する。

結論（該当モジュール）

- src/effects/dash.py — 環境変数 `PYX_USE_NUMBA` / `PYX_USE_NUMBA_DASH`（レガシー `PXD_USE_NUMBA_DASH` も可）
  - 役割: numba 経路の有効/無効を切替。
  - 既定: 未設定時は使用（"1"）。`"0"|"false"|"False"|"FALSE"` で無効化。
  - 実装参照: `src/effects/dash.py:48`（共通/個別/レガシーの順で判定）
- src/effects/collapse.py — 環境変数 `PYX_USE_NUMBA`
  - 役割: numba 経路の有効/無効を切替。
  - 既定: 未設定時は使用（"1"）。`"0"|"false"|"False"` で無効化。
  - 実装参照: `src/effects/collapse.py:412`（`os.environ.get("PYX_USE_NUMBA", "1")`）

補足

- 上記はいずれも「numba が導入されている場合に限り既定で有効」とし、環境変数で opt-out する設計。
- 現時点で shapes 配下に「numba のために環境変数を参照する」実装は見当たらない（numba 自体の使用はあり）。
- `NUMBA_*` 系の公式環境変数（例: `NUMBA_DISABLE_JIT`, `NUMBA_NUM_THREADS` など）を直接参照するコードは effects/shapes には存在しない。
