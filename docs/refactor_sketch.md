# src/api/sketch.py リファクタリング計画（ドラフト）

目的: 原則として機能削減は行わず、コード行数を削減し、不要な重複/分岐/例外捕捉を整理して明確で美しい実装へ単純化する。なお、合意により `midi_strict` は削除する（非互換）。

- 対象: `src/api/sketch.py`（現在 約556行）
- ゴール（段階目安）:
  - Phase 1: 機能非削減の簡素化（後方互換維持） 556→≈430行
  - Phase 2: 引き続き非削減の整理（責務分離/重複排除）≈430→≈380行
  - Phase R: `midi_strict` の削除（合意済みの限定的な非互換）
  - Phase X: 将来の破壊的簡素化（本計画の対象外、別合意時のみ）
- 完了条件（変更単位）: 変更ファイルに対して ruff/mypy/pytest が成功。必要時スタブ再生成。

---

## 1. 現状の肥大/複雑ポイント（要約）
- 設定/環境のフォールバック分岐が多い
  - `fps` の設定ファイルフォールバック: `util.utils.load_config()` 参照（例: src/api/sketch.py:78）
  - `midi_strict` の環境変数/設定フォールバック（例: src/api/sketch.py:210）
- MIDI 初期化とフォールバックの分岐・厳格モード（例: src/api/sketch.py:235）
- Parameter GUI 統合とスナップショット分岐（例: src/api/sketch.py:276）
- HUD 用メトリクス連携（shape/effect キャッシュ）と on_metrics コールバック（例: src/api/sketch.py:332, 359）
- 色の正規化（hex 文字列対応など）の頑健化（例: src/api/sketch.py:421）
- PNG/G-code の複数モードと非同期進捗監視/キャンセル（例: src/api/sketch.py:493 以降）

---

## 2. 方針
- 明確さ/単純さを最優先しつつ、機能は一切削減しない（後方互換を保持）。
- 既存の公開 API 名/引数/動作を維持。挙動差分は導入しない（ドキュメントの整合のみ修正）。
- 将来的な大幅簡素化は別計画（Phase X）として分離する。

---

## 3. リファクタリング案（機能非削減の段階的整理）

### Phase 1（非破壊・挙動不変の簡素化）
- [ ] 冗長な分岐/代入の削除（例: `draw_callable`/`worker_count` の同値分岐を1系統に統合、src/api/sketch.py:276 付近）
- [ ] MIDI 初期化のフォールバック/ログ処理を関数化して集約（strict 削除後の単一路線、src/api/sketch.py:231 以降）
- [ ] HUD キャッシュ連携のゲーティングを1箇所に集約（既定OFFは維持、src/api/sketch.py:319-341）
- [ ] 色正規化の最小化（現行の hex/RGBA サポートは維持、例外文言と clamp を簡素化、src/api/sketch.py:405-451）
- [ ] Keyイベント内の長大処理を関数へ分離（PNG 保存処理、G-code 進捗/キャンセルを `handle_save_png`/`handle_export_gcode` に抽出、src/api/sketch.py:470 以降）
- [ ] on_close の後始末を関数化し重複除去（存在チェックは必要最小限、src/api/sketch.py:558 付近）
- [ ] ドキュメント整合（Parameter GUI と workers の説明差分を修正、src/api/sketch.py:128 の docstring）

期待効果: 読みやすさ向上と重複削減で ≈-120〜-150行。

### Phase 2（非破壊・責務分離/見通し改善）
- [ ] 小規模ヘルパの外出し（本モジュール内トップレベル関数として）
  - [ ] `resolve_fps()`、`normalize_color()`、`build_projection()`
  - [ ] `init_midi()`（Null/実機/フォールバックの分岐をここで完結）
  - [ ] `make_gcode_export_handlers()`（進捗ポーリング含む）
- [ ] 型注釈と別名の整理（`RGBA = tuple[float,float,float,float]` など）
- [ ] 例外捕捉の縮約（広域 try/except→狭域化、ログ/ HUD 表示は維持）
- [ ] コメント/説明の簡潔化（Why/Trade-off を短く、重複を削除）

期待効果: 行数をさらに ≈-50〜-70行、可読性と保守性を向上。

---

## 3R. `midi_strict` の削除（合意済み・限定的非互換）

目的: エラーハンドリングの二重化（strict/非strict）を廃し、分岐・設定・環境依存を削減。ユーザー体験は「MIDI 初期化失敗時は常にフォールバック（警告通知）」に統一する。

- [ ] 公開 API 変更: `run_sketch` の引数から `midi_strict: bool | None = None` を削除
- [ ] 挙動: `use_midi=True` かつ初期化失敗時は常に NullMidi へフォールバックし継続（従来の非strict相当）。`SystemExit(2)` 経路を削除
- [ ] 環境変数: `PYXIDRAW_MIDI_STRICT` のサポートを削除
- [ ] ドキュメント: モジュール先頭 docstring と README/architecture.md 内の関連説明を更新（strict 記述の削除）
- [ ] 実装: strict 分岐の除去、`resolve_midi_strict()` 相当の補助関数は撤去
- [ ] 型/スタブ: `src/api/__init__.pyi` への影響がある場合はスタブ再生成（`python -m tools.gen_g_stubs`）
- [ ] テスト: strict を前提にした分岐テストがあれば削除/更新
- [ ] ロギング/HUD: 初期化失敗時の HUD メッセージと warning ログは維持

移行ノート（想定）:
- 既存コードで `run_sketch(..., midi_strict=True/False)` を与えると `TypeError: got an unexpected keyword argument 'midi_strict'` となるため、呼び出し側の該当引数を削除する。

### 3R-EDIT: 実編集箇所リスト（参照行は開始行）
- `src/api/sketch.py:43` 引数説明の `midi_strict` 行を削除
- `src/api/sketch.py:47` 環境変数 `PYXIDRAW_MIDI_STRICT` の記述を削除
- `src/api/sketch.py:75` 注意/制限の「厳格 MIDI モード」記述を削除/更新
- `src/api/sketch.py:84` `import os` を削除（未使用化）
- `src/api/sketch.py:129` `run_sketch` シグネチャから `midi_strict: bool | None = None,` を削除（カンマ/デフォルト調整）
- `src/api/sketch.py:173` docstring の `midi_strict` パラメータ説明を削除（173–175 付近）
- `src/api/sketch.py:205` `midi_strict is None` から始まる環境変数/設定フォールバックの解決ブロックを削除（205–218 付近）
- `src/api/sketch.py:247` 例外時の strict 分岐を削除し、常に警告→NullMidi フォールバックに統一（247–254 を再編）

関連ドキュメント/設定/テストの追随
- `architecture.md:153` `midi_strict=True` で致命扱いの説明を削除/更新（常時フォールバックに統一）
- `architecture.md:289` 設定での `midi.strict_default` 記述を削除
- `architecture.md:293` 環境変数 `PYXIDRAW_MIDI_STRICT` の説明を削除
- `configs/default.yaml:17` `midi.strict_default` キーを削除（未使用化）
- `tests/api/test_sketch_more.py:18` `test_runner_midi_strict_true_exits_when_mido_missing` を削除/改名・改修
- `tests/api/test_sketch_more.py:44` `midi_strict=False` を前提とする呼び出し箇所を削除/改修
- `tests/api/test_sketch_more.py:50` `test_runner_env_var_controls_midi_strict` を削除
- `tests/api/test_sketch_more.py:62` `midi_strict=None` 前提の呼び出しを削除/改修
- `reports/agents_audit_2025-10-17.md:12` 監査ノート内の `PYXIDRAW_MIDI_STRICT` 記述を削除/更新
- `src/api/__init__.pyi` スタブ再生成で `run_sketch` のシグネチャから `midi_strict` を除去（以下を実行）
  - `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

---

## 4. 非互換の扱い（今回の計画）
今回の計画で導入する非互換は `midi_strict` の削除のみ（合意済み）。その他は非破壊・挙動不変で進める。
（将来案としての破壊的簡素化は Phase X として別ドキュメント化する）

---

## 5. 作業手順（反復ループ）
1) Phase 1 を実装（冗長/重複/長大箇所の抽出・簡素化）
2) 変更ファイル限定の検証: `ruff check --fix src/api/sketch.py && black src/api/sketch.py && isort src/api/sketch.py && mypy src/api/sketch.py`
3) 関連スモークテストの最小実行（例: `pytest -q -m smoke` または対象テスト）
4) Phase 2 の責務分離（トップレベル関数化）を適用
5) ドキュメント整合（docstring/architecture.md の説明差分のみ）

---

## 6. 受け入れ基準（DoD）
- `src/api/sketch.py` の行数が段階目標に沿って削減されている（機能差分なし）
- 変更ファイルに対する `ruff/mypy/pytest` が成功
- 公開 API（`run_sketch`）の挙動に差分がない（MIDI/HUD/GUI/PNG/G-code/WorkerPool 維持）。ただし `midi_strict` の削除を除く

---

## 7. 確認したいこと（選択式）
- [ ] Phase 1（非破壊）を実施してよい
- [ ] Phase 2（非破壊の責務分離）まで進めてよい
- [ ] ドキュメント整合（workers × GUI の説明修正）を行ってよい
- [ ] `midi_strict` 削除の実装に着手してよい（呼び出し側修正は利用者対応）

---

メモ: 実装中に追加の簡素化候補や懸念が出た場合は、このファイルに追記して都度提案・相談します。
