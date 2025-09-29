# `engine.ui.parameters.window` 冗長レイヤ削除 計画

目的
- `src/engine/ui/parameters/window.py`（DPG 実装の薄い再エクスポート）を撤廃し、実装直参照に一本化してシンプル化する。

背景 / 現状
- 現状は `ParameterWindow` の実体が `src/engine/ui/parameters/dpg_window.py` にあり、
  `src/engine/ui/parameters/window.py` はそれを再エクスポートしている。
- 参照箇所:
  - コントローラ: `src/engine/ui/parameters/controller.py:12`
  - テスト: `tests/ui/parameters/test_dpg_mount_smoke.py:9`
- 設計ドキュメントではパッケージに `ParameterWindow` がある旨の説明があるが、再エクスポート前提ではない。

方針
- 将来の差し替え余地は不要という前提で、利用側を DPG 実装へ直結し、再エクスポートレイヤを削除する。
- 追加の再エクスポート（`__init__.py` 等）は行わない。

影響範囲
- import 経路の変更（2箇所）。
- 再エクスポートモジュールの物理削除（1ファイル）。
- ドキュメントの文言微調整（再エクスポート言及を避けるトーンに統一）。

実施手順（チェックリスト）
- [ ] import 切替: `src/engine/ui/parameters/controller.py`
      - 変更前: `from .window import ParameterWindow`
      - 変更後: `from .dpg_window import ParameterWindow`
- [ ] import 切替: `tests/ui/parameters/test_dpg_mount_smoke.py`
      - 変更前: `from engine.ui.parameters.window import ParameterWindow`
      - 変更後: `from engine.ui.parameters.dpg_window import ParameterWindow`
- [ ] ファイル削除: `src/engine/ui/parameters/window.py`
- [ ] ドキュメント更新: `architecture.md`
      - `ParameterWindow` の所在を「DPG 実装（`engine.ui.parameters.dpg_window`）」と明記（再エクスポートの含みを避ける）。

検証（編集ファイル優先）
- Lint/Format/Type（各ファイル単位）
  - `ruff check --fix src/engine/ui/parameters/controller.py tests/ui/parameters/test_dpg_mount_smoke.py`
  - `black src/engine/ui/parameters/controller.py tests/ui/parameters/test_dpg_mount_smoke.py && isort src/engine/ui/parameters/controller.py tests/ui/parameters/test_dpg_mount_smoke.py`
  - `mypy src/engine/ui/parameters/controller.py`
- テスト（対象限定）
  - `pytest -q tests/ui/parameters/test_dpg_mount_smoke.py`

成功条件（DoD）
- 参照先が `dpg_window.ParameterWindow` に統一され、`engine.ui.parameters.window` への参照がリポ内に存在しない。
- 変更ファイルに対する ruff/black/isort/mypy が成功。
- `tests/ui/parameters/test_dpg_mount_smoke.py` が緑（DPG 未導入環境ではスタブで通る）。
- `architecture.md` が現実の実装と矛盾しない。

ロールバック手順
- `src/engine/ui/parameters/window.py` を復活（元の再エクスポート実装を戻す）。
- `controller.py` / テストの import を元に戻す。

リスクと備考
- 外部コードが `engine.ui.parameters.window` を直接参照している場合に破壊的変更となるが、本リポは未配布のため許容。
- `dpg_window.py` は Dear PyGui 未導入環境向けのスタブを内包しており、テストの挙動は維持される見込み。

承認後の実行順序（提案）
1. import 切替（2箇所）
2. Lint/Format/Type（対象ファイル）
3. テスト（対象ファイル）
4. `window.py` 削除
5. `architecture.md` 更新
6. 仕上げの Lint（対象ファイル）

作業者メモ
- 変更は小さく、影響面は限定的。念のため ripgrep で `engine.ui.parameters.window` 検索を再実行して参照消滅を確認する。

