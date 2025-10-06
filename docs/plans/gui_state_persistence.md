# GUI パラメータの保存/復元（MIDI CC と同等の永続化）計画

この文書は、「MIDI コントローラの CC を `data/cc` に保存している」のと同様に、GUI で調整した値（Parameter GUI の override）も保存し、同じスクリプト実行時に復元するための設計と実装チェックリストである。破壊的変更は許容するが、今回は既存 CC 保存を壊さない最小追加でクリーンに実装する方針とする。

## ゴール / 非ゴール

- ゴール
  - GUI で調整したパラメータ値（override）を JSON で保存する。
  - 次回「同じスクリプト」を実行したときに override を復元し、GUI/実行に反映する。
  - 既存の CC 保存（`data/cc/<script>_<port>.json`）はそのまま動作させる。
- 非ゴール
  - CC と GUI の保存ファイルを即座に統合する（後述の代替案）。
  - 既存のパラメータ優先順位（「明示引数 > GUI > 既定値」）の変更。

## 仕様（保存/復元の動作）

- 保存対象
  - Parameter GUI に表示される「未指定（既定値採用）」の実引数が、GUI により上書きされた値（override）。
  - 数値（float/int/bool）、列挙（choices ありの enum）、ベクトル（長さ 3 または 4）。
  - 明示引数（provided）は保存対象外（現行仕様のまま）。
- 保存タイミング
  - アプリ終了時（`on_close`）に保存（MIDI CC 保存と同タイミング）。
- 復元タイミング/優先順位
  - `ParameterManager.initialize()` の初回トレース後、GUI ウィンドウ生成前にロードして `store.set_override()` で適用。
  - 実行時の優先順位は「明示引数 > 復元された GUI override > 既定値」を維持。
- ファイル配置/命名
  - 既定ディレクトリ: `data/gui/`
  - ファイル名: `<script_stem>.json`（例: `demo_shape_grid.json`）
  - 上書き: 設定 `parameter_gui.state_dir`（絶対/相対。未指定時は既定を使用）。
- フォーマット（JSON）
  - キーは Descriptor ID（`scope.name#index.param`）。値は実値（float/int/bool/enum/ベクトル）。
  - 例:
    ```json
    {
      "version": 1,
      "script": "demo/shape_grid.py",
      "saved_at": "2025-09-30T23:59:59Z",
      "overrides": {
        "effect.trim#0.amount": 0.35,
        "shape.circle#0.radius": 48.0,
        "effect.translate#1.offset": [10.0, 0.0, 0.0]
      }
    }
    ```
- 整数/真偽/列挙/ベクトルの扱い
  - JSON へはそのまま保存（列挙は文字列）。
  - べき等性確保のため、float は保存時に量子化（丸め）を行う（確定）。
    - 既定の量子化粒度: `1e-6`（環境変数 `PXD_PIPELINE_QUANT_STEP` で上書き可）。
    - `__param_meta__['step']`（または Descriptor の RangeHint/VectorRangeHint の step）がある場合はその粒度で丸める（ベクトルは成分ごと）。

## 設計

- 単純な責務分離
  - 新規モジュール `src/engine/ui/parameters/persistence.py` を追加（標準ライブラリのみ）。
    - `resolve_state_dir()`・`state_path_for_script()` のパス決定ヘルパ。
    - `load_overrides(script_path: str, store: ParameterStore) -> int`（適用件数を返す）。
    - `save_overrides(script_path: str, store: ParameterStore) -> Path`。
    - 量子化は Descriptor の RangeHint/VectorRangeHint の step（あれば）を優先し、無い場合は `1e-6`（`PXD_PIPELINE_QUANT_STEP` 反映）で四捨五入。
  - `ParameterManager` にロード/セーブを組み込む。
    - `initialize()` 内、初回トレース（descriptor 登録）直後に `load_overrides()` を呼ぶ（GUI マウント前）。
    - `shutdown()` 内で `save_overrides()` を呼ぶ。
  - `api/sketch.py` の `on_close` は現行の `parameter_manager.shutdown()` 呼び出しで保存まで到達（追加改修は最小）。
- マッピングの前提
  - Descriptor ID は `scope.name#index.param`。同一スクリプト・同一ロジックであれば安定。
  - 不整合（キーが見つからない/型が異なる）は安全側で無視し、可能な限り残りを適用。
  - 保存時の override 判定は公開 API のみで行う（`current_value(id) != original_value(id)` を採用し、内部状態へは直接アクセスしない）。
- 設定
  - `config.yaml` で `parameter_gui.state_dir` を任意指定可能（`io.cc_dir` と同様の解決）。
  - 未指定時は `CWD/data/gui`。

## 実装チェックリスト（作業単位）

- [x] `src/engine/ui/parameters/persistence.py` を追加（JSON 読書き + 量子化 + 例外安全）。
- [x] `ParameterManager.initialize()` にロード処理を追加（GUI 起動前に適用）。
- [x] `ParameterManager.shutdown()` にセーブ処理を追加（終了時に保存）。
- [x] `util.utils.load_config()` を用いた `parameter_gui.state_dir` の解決を実装。
- [x] 保存フォルダが存在しない場合に作成（`mkdir(parents=True, exist_ok=True)`）。
- [x] ベクトル/enum/bool/int/float のシリアライズ/デシリアライズ互換を確認。
- [x] 量子化: RangeHint/VectorRangeHint の step を優先し、無い場合は 1e-6（`PXD_PIPELINE_QUANT_STEP` 可）で丸め。
- [x] エラー時のフェイルソフト（読み込み失敗時は無視・新規起動、保存失敗はログのみ）。
– [ ] ドキュメント更新（本計画を README/architecture にリンク、AGENTS.md の運用項目は現状維持）。
- [ ] 最小テスト追加（単体）
  - [ ] 登録 →override 設定 → 保存 → 新規 Store へロード → 値一致。
  - [ ] provided 値は保存対象外であること（現行仕様の保持）。
  - [ ] ベクトル/enum/bool/int/float の往復保存。

## 受け入れ条件（DoD）

- `demo/shape_grid.py` などのサンプルで GUI を開き、いくつかの値を変更して終了 → 再起動で値が復元される。
- MIDI 有効/無効の別にかかわらず、終了時に CC と GUI がそれぞれ保存される（保存先は別ディレクトリ）。
- Lint/Format/Type/Test（変更ファイルに限定）がすべて緑。

## 代替案（情報共有のみ・今回は実装しない）

- CC/GUI を単一状態ファイルに統合
  - 例: `data/state/<script_stem>.json` に `{"cc": {"<port>": {...}}, "gui": {...}}` で同梱。
  - メリット: ファイル散在の低減、保存/読み込みの一元化。
  - デメリット: 既存 CC 実装（`MidiController`）の保存責務変更が大きく、互換性注意。
  - 将来移行時は `engine.io.controller.MidiController.save_cc()` を委譲化して段階移行する。

## 確定事項（ユーザー合意）

- 保存ディレクトリ: `data/gui` を使用（設定キー `parameter_gui.state_dir` で上書き可）。
- 保存タイミング: 終了時のみ（デバウンス等の自動保存は実装しない）。
- 量子化: `__param_meta__['step']`（RangeHint/VectorRangeHint の step を含む）優先、未指定は `1e-6`。
- リセット操作: 提供しない（不要）。

---

更新履歴

- 2025-10-06: 初版（計画・チェックリスト作成）。
- 2025-10-06: 方針確定に合わせて簡素化（終了時保存のみ、保存先確定、量子化/リセットの扱い明文化）。
