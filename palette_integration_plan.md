# palette 統合計画（確認用ドラフト）

目的: `src/palette` のカラー生成コアをドメイン層として整理しつつ、API から `C` を提供し、既存 parameter_gui に `palette` ヘッダを追加してその中でパラメータ操作できるようにする。ここでは実装前のチェックリストを提示する。

## やること（チェックリスト）

- [x] 配置/責務の確定: `src/palette` は計算コア（L1）として維持し、UI 向け補助は `engine/ui/palette/` 側へ分離、公開面は `api/palette.py` から `C` を再エクスポートする構成で進める。
- [x] API 設計: `C` オブジェクトのインターフェース（`__getitem__`, `__len__`, 更新フック、型/Docstring、スタブ生成対象）と初期化フローを定義する。
- [x] アーキテクチャ定義更新: `architecture.md` と `tests/test_architecture.py` の LAYER_MAP に `palette` を追加し、依存規約を明文化する。
- [x] データフロー設計: GUI 変更 → パレット生成 → ワーカー/描画への反映経路を cc と同型のスナップショット共有で実装する。
- [x] GUI 実装方針: 既存 parameter_gui に `palette` ヘッダを新設し、そのセクション内にパレット GUI（ラベル選択・色表示・エクスポート操作）を配置する。`run(..., enable_palette_gui=True)` フラグで有効/無効を切り替えつつ、Parameter GUI との起動タイミングを設計する。
- [ ] モジュール分割: `ui_helpers.py` 等 UI 寄りの機能を `engine/ui/palette` 側へ移すか再実装し、`src/palette` には純計算のみを残す（外部コピー互換は不要。必要なら `palette/USAGE.md` も更新）。
- [x] 公開 API/ドキュメント: `api/__init__.py`/`api/palette.py` への追加、使用例（draw 内での `C[0]` など）を README/USAGE に追記。
- [x] テスト計画: API レベル（`C` のアクセス/更新）、アーキテクチャテスト、必要に応じて GUI 用の最小スモーク（オプショナル）を用意。
- [x] スタブ同期: 公開 API 変更後に `tools/gen_g_stubs.py` 実行と同期テスト更新。

## 確認回答の反映
- C は単一パレットを扱う。
- GUI 起動は `enable_palette_gui` フラグで制御し、デフォルト True。
- パレット状態は永続化と config 連携を行う（保存/復元）。
- ワーカ伝搬は cc と同型のスナップショット経路を新設し、`draw` は `C[0]` を読むだけでよい。
- 外部コピー向け後方互換は不要。クリーン実装を優先し、必要なら `palette/USAGE.md` も更新する。
