# 大規模リファクタ計画: `draw(t)` + グローバル `api.cc`

目的: `draw` は `t` のみを受け取り、CC 値はユーザーが `from api import cc` で参照する。`cc[i]` は 0.0–1.0 の float を返し、MIDI のみで更新される（GUI は引数 UI のみ）。実装は破壊的変更を許容し、最小で美しい構成に収束させる。

想定効果:
- ユーザー関数の署名簡素化: `def draw(t: float) -> Geometry`
- CC 参照の一元化: `from api import cc; v = cc[1]`（`get()` は提供しない）
- GUI/MIDI の有無に関わらず同一コードが動作（GUI はパラメータ UI のみ）

---

## 設計方針（シンプルさ最優先）

- `api.cc` は「現在フレームの MIDI CC スナップショット」を返す読み取り専用プロキシ。
  - `cc[i] -> float`（未定義は 0.0）。`get()` は提供しない。
  - 実体は極薄のプロキシ（安全な 0.0 既定）。
- スナップショットの供給点は 1 箇所のみ:
  - `WorkerPool` が MIDI のスナップショットを取得し、`api.cc` にセットしてから `draw(t)` を呼ぶ。
- `draw(t)` 実行中は CC スナップショットを不変とする。
- パラメータ GUI は従来通り「関数名×引数名」ラベルに紐づく UI。GUI 値は `cc` へは混在させず、引数解決のみで適用。

### MIDI/GUI 同期ポリシー（重要）

- 片方向同期規則:
  - MIDI(cc) 変更: 描画値と GUI スライダーの双方を更新（即時ジャンプ）。
  - GUI 変更: 描画値のみ更新（cc 値は書き換えない）。
- スコープ: 変更が入った“そのパラメータのみ”を更新し、他のパラメータには影響しない。
- 優先順位: `midi > gui > original`。MIDI イベント到来時は GUI の値を上書きして追従させる。
- バインドの指定: ユーザーは draw 内で `param=cc[i]`（またはそれに準じる式）と書くことで「そのパラメータは CC#i にバインド」される。以後その CC の変化は当該パラメータと GUI にのみ反映。

---

## 変更概要（破壊的）

- API 変更: `user_draw(t, cc)` -> `user_draw(t)`（必須変更）
- 新規公開 API: `from api import cc`（`cc.get` 非提供）
- ランタイム連携: `WorkerPool` のみが `cc` に現在スナップショットを注入（GUI は関与しない）
- ドキュメント/サンプル/テストの署名更新

---

## アーキテクチャ（簡潔）

構成要素（責務最小）:
- api.cc: 現在フレームの MIDI CC を保持するプロキシ。`cc[i] -> CCBinding`（演算合成対応）。`set_snapshot()` と `raw()` を提供。
- WorkerPool: 毎フレーム `midi.snapshot()` を取得→`api.cc.set_snapshot()`→`user_draw(t)` を実行。
- ParameterManager: `user_draw` をラップし、`begin_frame()`→`set_inputs(t, api.cc.raw())`→`user_draw(t)` の順で実行。GUI ウィンドウの寿命管理。
- ParameterStore: 単一の真実源。`original/gui_override/midi_override` を保持し、購読者（GUI）へ通知。
- ValueResolver: 形状/エフェクト呼び出し前にパラメータ実値を解決。`CCBinding` を検出し、`cc_snapshot` から値取得→`store.set_override(..., source="midi")` を適用。

1 フレームのデータフロー:
- MIDI 変更 → WorkerPool が新スナップショットを `api.cc` に設定。
- `user_draw(t)` 内で shape/effect に渡す引数が `cc[i]`（またはその合成）なら Resolver が検出し、Store の該当パラメータに `midi_override` を設定→GUI スライダーがジャンプ。
- GUI 変更 → Controller が `store.set_override(..., source="gui")`。描画値は直ちに反映。ただし次の MIDI イベント到来で上書き（midi>gui）。
- スコープは常に「そのパラメータのみ」。他のパラメータへは波及しない。

抽象化方針:
- 追加バスやイベント層は導入しない。Store を単一の真実源に据え、`api.cc` は極小の読み取りプロキシに限定。
- CC→GUI 同期は Resolver→Store の `midi_override` に集約し、GUI は Store の購読で一律に追従。

---

## 実装タスク（チェックリスト）

1) API: `api.cc` の導入と公開
- [ ] 新規 `src/api/cc.py`: `CCProxy` 実装（`__getitem__` のみ・`raw` は任意）
- [ ] セッター: `set_snapshot(mapping: Mapping[int, float])`（フレーム毎に上書き）
- [ ] `src/api/__init__.py` に `from .cc import cc` を追加し、`__all__` に `"cc"` を追加
- [ ] `src/engine/ui/parameters/cc_binding.py`: `CCBinding` に演算子を実装（+,-,*,/ の左右両方）。演算は `map` を合成して新しい `CCBinding` を返す。

2) ランナー/ワーカ連携
- [ ] `src/engine/runtime/task.py`: （現状維持）`cc_state` は残す
- [ ] `src/engine/runtime/worker.py`:
  - [ ] 型: `draw_callback: Callable[[float], Geometry]` に変更
  - [ ] 子プロセス/インライン双方で `api.cc.set_snapshot(task.cc_state)` を `draw` 前に適用
  - [ ] `draw_callback(task.t)` 呼び出しへ変更
- [ ] `src/api/sketch.py`:
  - [ ] `user_draw` 型を `Callable[[float], Geometry]` に変更
  - [ ] `WorkerPool` の新署名に合わせる（ParameterManager 経由でも引数は `t` のみ）

3) パラメータ GUI 統合（cc は参照のみ）
- [ ] `src/engine/ui/parameters/manager.py`:
  - [ ] `__init__` 型を `Callable[[float], Geometry]` に変更
  - [ ] `initialize()` のトレース呼び出しを `self._user_draw(0.0)` に変更
  - [ ] `draw(t)` に変更し、`runtime.begin_frame(); runtime.set_inputs(t, api.cc.raw())` を実施後、`self._user_draw(t)` を呼ぶ（GUI 値は Store から適用、cc は api 側スナップショットを参照）

4) サンプル/エントリ修正
- [ ] `main.py`/`main2.py`/`demo/*.py`/`effect.py`/`shape.py` を `def draw(t)` 化し、`from api import cc` 参照に変更

5) スタブ/公開 API 整合
- [ ] `tools/gen_g_stubs.py`:
  - [ ] 生成内容に `cc: Any` または専用 `CC` 型を追加（`get` を示唆しない）
  - [ ] `__all__` に `"cc"` を追加
- [ ] スタブ再生成: `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

6) テスト更新（編集ファイル優先）
- [ ] `tests/api/test_sketch_init_only.py`: `_draw(t: float) -> Geometry`
- [ ] `tests/api/test_sketch_more.py`: `_dummy_draw(t: float)` に変更
- [ ] `from api import cc` を使う最小参照を 1 箇所追加（存在確認）
- [ ] `tests/ui/parameters/test_cc_binding_vector.py` など CCBinding 依存は仕様変更に合わせて更新/無効化
 - [ ] 新規: MIDI→GUI 同期の単体（MIDI 変更で GUI 値がジャンプ）
 - [ ] 新規: GUI→描画のみの単体（GUI 変更は cc 値を書き換えない）

7) ドキュメント整合
- [ ] `architecture.md`: `draw(t, cc)` → `draw(t)`、`from api import cc` を明示（該当行を更新）
- [ ] `docs/proposals/*` の参照を現行設計へ注記（必要最小限）

8) ビルド/高速チェック（変更ファイル限定）
- [ ] Lint: `ruff check --fix {changed_files}`
- [ ] Format: `black {changed_files} && isort {changed_files}`
- [ ] Type: `mypy {changed_files}`
- [ ] Test: `pytest -q -k sketch -q` もしくは対象ファイル直指定

完了条件（DoD）
- [ ] 変更ファイルに対する `ruff/black/isort/mypy/pytest` 緑
- [ ] `api/__init__.pyi` が生成内容と一致（`tests/stubs/*` 緑）
- [ ] サンプル `python main.py` が実行可能（手元実行）

---

## 具体的インターフェース案

- `src/api/cc.py`
  - `class _CCProxy`（Mapping は継承しない）:
    - `__getitem__(i: int) -> float-like | CCBinding`: 未定義は 0.0（`get` は提供しない）
    - `raw() -> dict[int, float]`（任意、デバッグ用）
  - 内部実装:
    - モジュール内 `_snapshot: dict[int, float]` を保持
    - `set_snapshot(m: Mapping[int, float]) -> None` で差し替え
    - スレッド/プロセス毎に独立（Worker がフレーム毎に設定）

実装メモ（同期の成立条件）:
- `cc[i]` は「float 互換」でありつつ、パラメータに渡された際は「CC バインディング」として解釈される必要がある。
  - 具体案: `cc[i]` は CCBinding を返し、`__radd__/__rmul__/__add__/__mul__/__sub__/__truediv__` を実装して、合成関数（map）を累積した新しい CCBinding を返す。
  - `float(cc[i])` 明示キャスト時はバインディング情報は失われる（その場の値を使用）。
- Resolver は CCBinding を検出し、`cc_snapshot`（ParameterManager が `api.cc.raw()` から供給）を利用して値を取得、該当 Descriptor に `store.set_override(..., source="midi")` を適用。
- GUI 側は `ParameterStore` の値を購読し、midi>gui 優先でスライダーを更新する（MIDI イベントで即時ジャンプ）。

備考:
- 代入 API（`cc[i] = x`）は提供しない（更新は MIDI 経由）。
- 0.0–1.0 正規化は供給側で保証。`cc` は変換を行わない。

---

## 互換性・移行（破壊的変更）

- 旧: `def draw(t, cc)` → 新: `def draw(t)`
- 旧: `cc` は引数 → 新: `from api import cc`（グローバル）
- 例（変更前→変更後）
  - 前: `amp = 0.2 + 0.8 * cc.get(1, 0.0)`
  - 後: `from api import cc; amp = 0.2 + 0.8 * cc[1]`

---

## リスクと簡易対策

- スタブ/`__all__` 不整合 → 生成スクリプト更新 + テストで検出
- 並行実行時の CC 競合 → `WorkerPool` が `set_snapshot()` を `draw()` 前に設定、実行中は不変
- GUI 由来の CC 混在 → 仕様として切り離し（GUI は引数 UI のみ）

---

## 開発・検証メモ（編集ファイル限定で回す）

- ruff/black/isort/mypy は変更ファイルに限定
- スタブは最後に再生成→差分ゼロ確認
- テストは `tests/api/test_sketch_*` を優先的に緑化

---

## オープンクエスチョン（要確認）

- `cc` の型スタブは `Protocol` で `__getitem__` のみを公開するか、`Any` とするか
- `api.cc.raw()` をドキュメント化するか（デバッグ用途に限定）

---

以上の計画で進めてよいか確認してください。OK なら、順に実装し、完了項目にチェックを付けながら進捗を可視化します。
