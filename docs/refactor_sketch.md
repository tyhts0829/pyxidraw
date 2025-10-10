# src/api/sketch.py リファクタリング計画（再同期版）

目的: 原則として機能削減を行わずに、不要な重複/分岐/例外捕捉を整理して明確で美しい実装へ単純化する。実装の進捗に合わせて計画を再同期する。

- 対象: `src/api/sketch.py`（現在 約715行）
- ゴール（段階目安）:
  - Phase 1: 機能非削減の簡素化（後方互換維持） 715→≈570行
  - Phase 2: 引き続き非削減の整理（責務分離/重複排除）≈570→≈500行
  - Phase R: `midi_strict` の削除（限定的な非互換、要合意）
  - Phase X: 将来の破壊的簡素化（本計画の対象外、別合意時のみ）
- 完了条件（変更単位）: 変更ファイルに対して ruff/mypy/pytest が成功。必要時スタブ再生成。

---

## 1. 現状の概況（実装確認の要約）
- 設定/環境のフォールバック
  - `fps`: `util.utils.load_config()` から安全にフォールバック（未設定時は 60）。
  - `midi_strict`: 環境変数（`PYXIDRAW_MIDI_STRICT`）/設定（`midi.strict_default`）で解決。
- MIDI 初期化
  - `use_midi=True` で実機を試行。失敗時は strict のみ致命、通常は Null 実装へフォールバック。
- Parameter GUI
  - `engine.ui.parameters` を `ParameterManager` で統合。GUI 併用時もワーカ併用可能（スナップショット注入）。
- HUD/キャッシュメトリクス
  - `WorkerPool.metrics_snapshot` と `StreamReceiver(on_metrics)` をゲート付きで連携（`HUDConfig.show_cache_status`）。
- 色の正規化
  - `util.color.normalize_color` を採用し、背景未指定時は設定→既定白。線色未指定時は設定→背景輝度で黒/白を自動選択。
- エクスポート（PNG/G-code）
  - PNG: 画面直書き保存とオフスクリーン高解像度保存の 2 モード。
  - G-code: `ExportService` + 非同期ジョブ + 進捗ポーリング + キャンセルを実装。

---

## 2. 方針
- 明確さ/単純さを最優先しつつ、機能は一切削減しない（後方互換を保持）。
- 既存の公開 API 名/引数/動作を維持。挙動差分は導入しない（ドキュメントの整合のみ修正）。
- 将来的な大幅簡素化は別計画（Phase X）として分離する。

---

## 3. リファクタリング案（機能非削減の段階的整理）

### Phase 1（非破壊・挙動不変の簡素化）
- [ ] 冗長な分岐/代入の削除（例: `draw_callable`/`worker_count` の同値分岐を1系統に統合）
- [ ] MIDI 初期化のフォールバック/ログ処理を関数化して集約（現状はインライン）
- [x] HUD キャッシュ連携のゲーティングを集約（`metrics_snapshot`/`on_metrics` を条件付きで接続）
- [x] 色正規化の最小化（`util.color.normalize_color` を採用、背景輝度に応じた線色自動選択）
- [ ] Key イベント内の長大処理を関数へ分離（PNG/G-code を `handle_*` へ抽出）
- [ ] `on_close` の後始末を関数化（安全な冪等化）
- [ ] ドキュメント整合（docstring の workers×GUI 説明を現実に合わせて修正）

期待効果: 読みやすさ向上と重複削減で ≈-140 行。

### Phase 2（非破壊・責務分離/見通し改善）
- [ ] 小規模ヘルパの外出し（本モジュール内トップレベル関数として）
  - [ ] `resolve_fps()`、`build_projection()`
  - [ ] `init_midi()`（strict 分岐の有無に関わらずここで完結）
  - [ ] `make_gcode_export_handlers()`（進捗ポーリング含む）
- [ ] 型注釈と別名の整理（`RGBA = tuple[float,float,float,float]` など）
- [ ] 例外捕捉の縮約（広域 try/except→狭域化、ログ/HUD 表示は維持）
- [ ] コメント/説明の簡潔化（Why/Trade-off を短く、重複を削除）

期待効果: 行数をさらに ≈-70 行、可読性と保守性を向上。

---

## 3R. `midi_strict` の削除（限定的非互換・要合意）

目的: エラーハンドリングの二重化（strict/非strict）を廃し、分岐・設定・環境依存を削減。ユーザー体験を「MIDI 初期化失敗時は常にフォールバック（警告通知）」に統一する。

- 現状: `midi_strict` は存続しており、環境変数/設定から解決。strict 時のみ致命終了。
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
1) Phase 1 を実装（冗長/重複/長大箇所の抽出・簡素化）
2) 変更ファイル限定の検証: `ruff check --fix src/api/sketch.py && black src/api/sketch.py && isort src/api/sketch.py && mypy src/api/sketch.py`
3) 関連スモークテストの最小実行（例: `pytest -q -m smoke` または対象テスト）
4) Phase 2 の責務分離（トップレベル関数化）を適用
5) ドキュメント整合（`src/api/sketch.py` の docstring と `architecture.md`、必要に応じて README）

---

## 6. 受け入れ基準（DoD）
- `src/api/sketch.py` の行数が段階目標に沿って削減されている（機能差分なし）
- 変更ファイルに対する `ruff/mypy/pytest` が成功
- 公開 API（`run_sketch`）の挙動に差分がない（MIDI/HUD/GUI/PNG/G-code/WorkerPool 維持）。ただし `midi_strict` の削除を除く

---

## 7. 確認したいこと（選択式）
- [ ] Phase 1（非破壊）を実施してよい
- [ ] Phase 2（非破壊の責務分離）まで進めてよい
- [ ] ドキュメント整合（docstring/architecture の修正）を行ってよい
- [ ] `midi_strict` 削除の実装に着手してよい（呼び出し側修正は利用者対応）

---

補足（実装との整合メモ）
- workers × Parameter GUI: 現行実装は「併用可能」。GUI の値は `WorkerPool` へスナップショットで注入されるため、シングル固定は不要（docstring を修正する）。
- 色: `util.color.normalize_color` を単一路線として採用。線色は背景輝度からの自動決定を実装済み。
- HUD キャッシュ: `metrics_snapshot`/`on_metrics` の両方にゲートを設け、HUD 無効時のコストを抑制。

メモ: 実装中に追加の簡素化候補や懸念が出た場合は、このファイルに追記して都度提案・相談します。
