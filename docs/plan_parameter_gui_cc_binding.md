# Parameter GUI: CCバインド最小実装 計画（チェックリスト）

どこで: engine.ui.parameters / api 層（UI+実行時の最小改修）
何を: 各パラメータ行に「CC番号」入力を追加し、設定された整数CCの値（0..1）で実行時のパラメータ値を制御可能にする。
なぜ: MIDIコントローラのノブから直接パラメータを操作するため。GUI表示値は自動で上書きせず、優先順位「明示引数 > GUI > 既定値」を維持する。

## スコープ（最小）
- 対象型: 数値スカラ（float/int）のみ。bool/enum/string/vector は対象外。
- レンジ: `RangeHint`（min/max/step）があればそれを使用、無ければ 0..1。
- ステップ: 表示用のみ（キャッシュ鍵の量子化は既存処理に委譲）。
- 永続化: CCバインドはセッション限定（保存しない）。必要なら後続で拡張。

## 仕様（ユーザー視点）
- 各数値スライダーの右に小さな「CC」入力ボックスが追加される。
  - 空欄: 未バインド。
  - 0..127 の整数: 指定CCにバインド。該当CCの現在値（0..1）がパラメータの実レンジに線形スケールされ、実行に用いられる。
- GUIのスライダー表示値はMIDIで自動更新されない（現行仕様「MIDI→GUIの自動上書きなし」を維持）。
- 優先順位は「明示引数 > GUI(=CC含む) > 既定値」。明示引数が与えられた場合はCCは無視される。

## 設計（最小差分）
- ParameterStore に CCプロバイダ（現在スナップショット）とバインディング機能を追加。
  - `set_cc_provider(provider: Callable[[], Mapping[int, float]])`（0..1正規化マップを返す）
  - `bind_cc(param_id: str, index: int | None)` / `cc_binding(param_id: str) -> int | None`
  - `cc_value(index: int) -> float`（プロバイダ無し/未定義は 0.0）
- ParameterValueResolver のスカラ解決（default採用時）で、CCバインドがあればCC値をレンジへスケールして返す。
  - float: `min + (max-min) * cc`
  - int: 上記を四捨五入
  - bool/enum/string/vector: 対象外（スキップ）
- スナップショット（ワーカ用）に CC 由来の実値を同梱。
  - `extract_overrides(store)` が「GUI override」に加えて「CCバインドされたスカラの実値」を入れる。
  - SnapshotRuntime は「未指定引数にのみ適用」なので、明示引数優先の原則を維持。
- GUI（Dear PyGui）: スライダー横に CC番号入力を追加。編集で `store.bind_cc(...)` を呼ぶ。
- CCプロバイダの注入は API 層（`api/sketch.py`）で `parameter_manager.store.set_cc_provider(api.cc.raw)` を呼ぶ（UI層から api.* へ依存しない）。

## 実装タスク（チェックリスト）
- [ ] Store: CCプロバイダ/バインドの追加（`src/engine/ui/parameters/state.py` の `ParameterStore`）
  - [ ] `_cc_provider: Callable[[], Mapping[int, float]] | None` と `_cc_bindings: dict[str, int]` を保持
  - [ ] `set_cc_provider()`/`bind_cc()`/`cc_binding()`/`cc_value()` を実装
- [ ] Resolver: スカラ default 解決でCC適用（`src/engine/ui/parameters/value_resolver.py`）
  - [ ] `_register_scalar()` 経由で登録後、`store.cc_binding(desc.id)` を確認
  - [ ] バインド有りなら `RangeHint`（無ければ0..1）へ線形スケールした値を返す（intは丸め）
  - [ ] provided（明示引数）経路は従来通り素通し
- [ ] Snapshot: CC実値の同梱（`src/engine/ui/parameters/snapshot.py`）
  - [ ] `extract_overrides()` で各Descriptorについて、バインド済みスカラなら実レンジにスケールした値を `overrides` に追加
  - [ ] 既存のGUI overrideは従来通り差分のみ
- [ ] GUI: CC番号入力の追加（`src/engine/ui/parameters/dpg_window.py`）
  - [ ] `_create_int()`/`_create_float()` 内で、スライダーの右に `dpg.add_input_text(tag=f"{desc.id}::cc")`
  - [ ] コールバック `_on_cc_binding_change(...)` を追加：空→unbind、数字→0..127にクランプして `store.bind_cc()`
  - [ ] 初期表示は既存バインドがあればその数値、無ければ空
- [ ] API: プロバイダ注入（`src/api/sketch.py`）
  - [ ] `parameter_manager = ParameterManager(user_draw)` 生成後に `from api import cc as _cc; parameter_manager.store.set_cc_provider(_cc.raw)` を呼ぶ
- [ ] 手動検証（smoke）
  - [ ] デバイス無し（NullMidi）環境でUIが表示されること
  - [ ] 数値パラメータのCC欄に整数を入れてもクラッシュしないこと
  - [ ] 実行時にCCスナップショットを差し替えると（デバイス or モック）、パラメータの見た目は維持されつつ描画が変化すること
- [ ] 高速チェック（変更ファイルに限定）
  - [ ] ruff/mypy（対象ファイルのみ）
  - [ ] 既存最小テストのスポット実行（例: `pytest -q tests/ui/parameters/test_runtime.py::test_parameter_runtime_registers_default_shape_parameters`）

## 非対応/先送り（必要なら別PR）
- ベクトル（3/4成分）へのCCバインド（単一CCで全成分、成分別など）
- CCバインドの永続化（`persistence.py` 拡張）
- CCバインド有効時のGUIスライダー表示値の追従（現仕様は自動上書き禁止）
- bool/enum/string への対応

## リスク/注意
- レイヤ境界: UI層から api への依存は避け、プロバイダは API 層で注入する。
- 既存の優先順位を壊さない: provided（明示引数）は最優先でCC適用対象外。
- レンジ不在時は 0..1 とみなす（既定）。
- 量子化/キャッシュ鍵は従来の `params_signature` に委譲（Effectsは量子化後の値で実引数にも渡る）。

## 受け入れ条件（DoD）
- 数値スカラ行にCC入力が出現し、整数バインドで実行時にMIDIで制御できる。
- 明示引数 > GUI(CC) > 既定値 の優先順位を満たす。
- 変更ファイルに対する `ruff`/`mypy`/（必要箇所の）`pytest -q` が緑。
- 追加依存なし、ネットワークアクセスなし。

## 相談事項（要指示）
- ベクトルへの対応は現時点で不要ですか？（最小実装としてスコープ外のままで良いか）
- CCバインドの永続化は必要ですか？（必要なら保存/復元フォーマットを追記します）

---
このチェックリストで問題なければ実装に着手します。修正・追加のご要望があれば指示ください。

