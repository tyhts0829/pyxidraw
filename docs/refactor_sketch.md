# src/api/sketch.py リファクタリング計画（現行実装に同期）

目的: 機能削減を行わずに、重複/分岐/例外捕捉を整理し、明確でシンプルな実装に保つ。実装の進捗に合わせて本計画を再同期する。

- 対象: `src/api/sketch.py`（現在 約735行）
- ゴール（段階目安・更新版）:
  - Phase 1: 機能非削減の簡素化（後方互換維持） 完了
  - Phase 2: 非破壊の責務分離/見通し改善（最小限の外出し）
  - Phase R: `midi_strict` の削除（限定的な非互換、要合意）
  - Phase X: 将来の破壊的簡素化（別合意時のみ）
- 完了条件（変更単位）: 変更ファイルに対して ruff/mypy/pytest が成功。必要時スタブ再生成。

---

## 1. 現状の概況（実装確認の要約）
- 設定/環境のフォールバック
  - `fps`: `util.utils.load_config()` から安全にフォールバック（未設定時は 60）。
  - `midi_strict`: 環境変数（`PYXIDRAW_MIDI_STRICT`）/設定（`midi.strict_default`）で解決（存続）。
- MIDI 初期化
  - `_setup_midi()` に集約。`use_midi=True` で実機を試行。失敗時は strict のみ致命、通常は Null 実装へフォールバック。
- Parameter GUI
  - `ParameterManager` を使用。GUI 併用時もワーカ併用可能（スナップショット注入）。色系は保存値/変更をメインスレッドへ安全に反映。
- HUD/キャッシュメトリクス
  - `metrics_snapshot` と `on_metrics` を `HUDConfig.show_cache_status` でゲート。`MetricSampler`/`OverlayHUD` を導入済み。
- 色の正規化
  - `util.color.normalize_color` を単一路線で採用。背景未指定時は設定→既定白。線色未指定時は背景輝度に応じて黒/白を自動選択。
- エクスポート（PNG/G-code）
  - PNG: オフスクリーン高解像度保存（Shift+P）と画面バッファ保存（P）を実装。
  - G-code: `ExportService` による非同期ジョブ。進捗ポーリング/完了・失敗表示/キャンセル（Shift+G）対応。
- 依存初期化の回避
  - `init_only=True` で `pyglet`/`moderngl` を import せず早期 return（テストで担保）。

---

## 2. 方針
- 明確さ/単純さを最優先しつつ、機能は一切削減しない（後方互換を保持）。
- 既存の公開 API 名/引数/動作を維持。挙動差分は導入しない（ドキュメントの整合のみ修正）。
- 将来的な大幅簡素化は別計画（Phase X）として分離する。

---

## 3. リファクタリング案（機能非削減の段階的整理）

### Phase 1（非破壊・挙動不変の簡素化）[完了]
- [x] 冗長な分岐/代入の削除（`draw_callable`/`worker_count` の同値分岐を一本化）
- [x] MIDI 初期化のフォールバック/ログ処理を関数化（`_setup_midi`）
- [x] HUD キャッシュ連携のゲーティング（`metrics_snapshot`/`on_metrics` を条件付き接続）
- [x] 色正規化の単純化（`normalize_color` 採用、背景輝度に応じた線色自動選択）
- [x] キーイベント処理の関数抽出（PNG/G-code の `_handle_save_png`/`_start_gcode_export` など）
- [x] `on_close` の冪等クリーンアップ化
- [x] ドキュメント整合（docstring の workers×GUI 説明を修正）

実績: 読みやすさ向上と重複削減を達成（行数は外部機能追加に伴い総量は維持）。

### Phase 2（非破壊・責務分離/見通し改善）
- [ ] 小規模ヘルパの外出し（必要最小限のみ）
  - [ ] `resolve_fps()`、`build_projection()` の切り出し（再利用/テスト容易化）
  - [ ] G-code エクスポート関係のハンドラ生成（`make_gcode_export_handlers()`）
- [ ] 型エイリアスの導入（例: `RGBA`）と局所的 `try/except` への縮約
- [ ] コメントの簡潔化（Why/Trade-off を短く、重複削除）

期待効果: 見通し向上とテスト容易化（挙動不変）。

---

## 3R. `midi_strict` の削除（限定的非互換・要合意）

目的: エラーハンドリングの二重化（strict/非strict）を廃し、分岐・設定・環境依存を削減。ユーザー体験を「MIDI 初期化失敗時は常にフォールバック（警告通知）」に統一する。

- 現状: `midi_strict` は存続しており、環境変数/設定から解決。strict 時のみ致命終了（テストで担保）。
- 変更案（合意後に実施）:
  - [ ] 公開 API 変更: `run_sketch` の引数から `midi_strict` を削除
  - [ ] 挙動: 失敗時は常に NullMidi へフォールバック（`SystemExit(2)` 経路を削除）
  - [ ] 環境変数/設定: `PYXIDRAW_MIDI_STRICT` と `midi.strict_default` を撤廃
  - [ ] ドキュメント: `src/api/sketch.py` の docstring と `architecture.md` の該当記述を更新
  - [ ] スタブ: `src/api/__init__.pyi` の再生成（`python -m tools.gen_g_stubs`）
  - [ ] テスト: strict 前提のテスト削除/更新
  - [ ] ロギング/HUD: フォールバック時の警告は維持

---

## 4. 非互換の扱い（今回の計画）
今回の計画で導入し得る非互換は `midi_strict` の削除のみ（要合意）。その他は非破壊・挙動不変で進める。
（将来案としての破壊的簡素化は Phase X として別ドキュメント化する）

---

## 5. 作業手順（反復ループ）
1) Phase 2 の責務分離（トップレベル関数化の最小適用）
2) 変更ファイル限定の検証: `ruff check --fix src/api/sketch.py && black src/api/sketch.py && isort src/api/sketch.py && mypy src/api/sketch.py`
3) 関連スモークテストの最小実行（例: `pytest -q -m smoke` または対象テスト）
4) ドキュメント整合（`src/api/sketch.py` の docstring と `architecture.md`、必要に応じて README）

---

## 6. 受け入れ基準（DoD）
- 変更ファイルに対する `ruff/mypy/pytest` が成功
- 公開 API（`run_sketch`）の挙動に差分がない（MIDI/HUD/GUI/PNG/G-code/WorkerPool 維持）。
- Phase R 実施時のみ `midi_strict` の削除に伴うスタブ・テスト整備が完了

---

## 7. 確認したいこと（選択式）
- [ ] Phase 2（非破壊の責務分離）を実施してよい
- [ ] ドキュメント整合（docstring/architecture の修正）を行ってよい
- [ ] `midi_strict` 削除（Phase R）に着手してよい（呼び出し側修正は利用者対応）

---

補足（実装との整合メモ）
- workers × Parameter GUI: 現行実装は「併用可能」。GUI の値は `WorkerPool` へスナップショットで注入されるため、シングル固定は不要（docstring は更新済み）。
- 色: `util.color.normalize_color` を採用。線色は背景輝度からの自動決定を実装済み。GUI からの変更はメインスレッドへスケジュールして適用。
- HUD キャッシュ: `metrics_snapshot`/`on_metrics` の両方にゲートを設け、HUD 無効時のコストを抑制。表示順はエフェクト→シェイプ。

メモ: 追加の簡素化候補や懸念が出た場合、本ファイルに追記して提案・相談する。
