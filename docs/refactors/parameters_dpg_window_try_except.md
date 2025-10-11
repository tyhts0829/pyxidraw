# dpg_window.py try/except 縮減リファクタ チェックリスト

目的: Dear PyGui 依存の UI 実装（`src/engine/ui/parameters/dpg_window.py`）に散在する広域 `try/except` を整理し、必要十分な堅牢さを保ちながら、読みやすくシンプルなコードへ改善する。

---

## 背景 / 現状
- UI 境界（Dear PyGui / pyglet / ユーザ設定）に合わせ、過剰に広い `except Exception:` が多用されている。
- 期待動作としてのフォールバック（例: 非対応属性のスキップ）と、想定外の失敗が混在し、ログや可読性が低下。
- 同様の防御コード（色正規化や `set_value` 例外抑止）が重複。

## スコープ
- 対象: `src/engine/ui/parameters/dpg_window.py`
- 非対象: 他モジュールの例外設計、UI 機能追加、API 変更。

## 方針（原則）
- 例外は種類を限定（`AttributeError`/`TypeError`/`ValueError`/`KeyError`）。広域 `except Exception:` は原則撤廃。
- 外部境界（pyglet import/DPG API/ユーザ設定読み込み）のみフェイルソフト。内部は事前検証で回避。
- 重複する防御はヘルパに集約し、呼び出し側の try/except を削減。
- フォールバックは `debug/warning`、想定外は `exception` でログレベルを分離。

## 受入基準（DoD）
- 変更箇所に対する `ruff/black/isort/mypy/pytest` が成功。
- `tests/ui/parameters/test_dpg_mount_smoke.py` 緑。
- 機能（生成/表示切替/終了/基本同期）は現状維持。
- ログが過剰に noisy にならず、想定外のみ `exception` が残る。

---

## 実施チェックリスト

- [ ] 1. try/except 分布の棚卸し（関数・行番号別の意図確認メモを追記）
- [ ] 2. ヘルパ導入で防御の集約
  - [ ] 2-1. `_safe_norm(value: Any, default: tuple[float,float,float,float]) -> tuple[float,float,float,float]`
        - 役割: 色正規化の例外吸収と既定値フォールバックを一箇所に集約
  - [ ] 2-2. `_supports_stretch_columns() -> bool`
        - 役割: `init_width_or_weight` サポート可否を事前分岐（`_add_two_columns` の try 除去）
  - [ ] 2-3. `_dpg_policy(policy_names: list[str]) -> Any | None`
        - 役割: `mvTable_Sizing*` の存在確認を一元化

- [ ] 3. 広域例外の撤去/限定（関数別）
  - [ ] 3-1. `ParameterWindow.set_visible`
        - 広域 `except` 撤去。状態ガードと DPG 呼び出しのみ。必要時 `RuntimeError` 等に限定
  - [ ] 3-2. `ParameterWindow.close`
        - 広域 `except` 撤去。`_stop_driver()` の内部のみ防御。`dpg.destroy_context()` は原則 fail-fast
  - [ ] 3-3. `ParameterWindow.mount`
        - 広域 `except` 撤去。`_build_grouped_table` の内部に責務を寄せる
  - [ ] 3-4. `_build_root_window`
        - トップレベルの広域 `except` 撤去。`build_display_controls` のみ局所防御（ログは warning）
  - [ ] 3-5. `build_display_controls`
        - 色取得/正規化の try 重複を `_safe_norm` で置換
        - `store.current_value/original_value` の取得は例外発生時のみ限定捕捉（`KeyError` 等）
  - [ ] 3-6. `force_set_rgb_u8`
        - 事前検証（長さ/範囲）で例外回避。例外は `ValueError` に限定
  - [ ] 3-7. `_add_two_columns`
        - `_supports_stretch_columns()` に置き換え。広域 `except` 撤去
  - [ ] 3-8. `_label_value_ratio`
        - 例外不要（単純な clamp のみ）
  - [ ] 3-9. `_on_widget_change`
        - ベクトル処理の例外を `TypeError/ValueError` に限定し、想定外は再送出または `logger.exception`
  - [ ] 3-10. `_on_store_change`
        - `normalize_color` の失敗のみ限定捕捉し、他は fail-fast。`does_item_exist` 後の `set_value` は原則ノー try
  - [ ] 3-11. `_apply_default_styles` / `_apply_styles_from_config` / `_apply_colors_from_config`
        - DPG 属性存在は事前確認。適用時の失敗は `TypeError/ValueError` のみ捕捉し warning
  - [ ] 3-12. `_to_dpg_color`
        - 例外限定＋バリデーション強化。無効値は `None` を返し呼び出し側がスキップ
  - [ ] 3-13. `_start_driver` / `_stop_driver`
        - 外部境界として現状維持（import/unschedule/stop 周りのみ防御）。ネストした広域 `except` は必要最小限へ縮小

- [ ] 4. ログの粒度調整
  - [ ] 4-1. 期待フォールバック（非対応属性/存在しないポリシー）は `debug` or `warning`
  - [ ] 4-2. 想定外の失敗は `logger.exception` を維持

- [ ] 5. ビルド/テスト（編集ファイル限定）
  - [ ] 5-1. `ruff check --fix src/engine/ui/parameters/dpg_window.py`
  - [ ] 5-2. `black src/engine/ui/parameters/dpg_window.py && isort src/engine/ui/parameters/dpg_window.py`
  - [ ] 5-3. `mypy src/engine/ui/parameters/dpg_window.py`
  - [ ] 5-4. `pytest -q tests/ui/parameters/test_dpg_mount_smoke.py`

- [ ] 6. 確認と微調整
  - [ ] 6-1. ログメッセージの過不足確認（フォールバック時に noisy でないこと）
  - [ ] 6-2. 例外伝播の妥当性（fail-fast させたい箇所が握り潰されていない）

---

## 追加の確認事項（要ご判断）
- 非 GUI 環境（Dear PyGui 未導入）の扱い: 現行はテスト側で `importorskip` 済み。DPG 未導入時のモジュール import 失敗は想定通りで良いか。
- `_on_store_change` のログレベル: 頻度が高い箇所のため、失敗ログを `warning` に抑制するか、`exception` を維持するか。
- テスト追加の要否: 内部ヘルパ（`_safe_norm`, `_supports_stretch_columns`）は私用メソッドだが、軽いユニットテストを追加するか。

---

## ロールバック指針
- UI 初期化や可視切替で新たな例外が観測された場合、該当箇所の `except` を一時的に復帰（限定例外での復帰を優先）。
- ログが不足する場合は、`warning` → `exception` へ段階的に引き上げ。

---

更新履歴
- v0: 初版（チェックリストと方針の提示）。

