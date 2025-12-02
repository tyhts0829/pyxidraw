# Parameter GUI 動的パラメータ保護計画

## 背景
- `docs/spec/parameter_gui_show_all_params_plan.md` に従い、明示指定パラメータも GUI へ登録した結果、`sketch/251203.py` 内の `E.displace(t_sec=t * 0.1)` のような時間依存パラメータが GUI 側の初期値で即座に上書きされ、`t` によるモジュレーションが効かなくなった。
- 原因は `ParameterWindowContentBuilder` が DPG ウィジェット生成時に `callback=_on_widget_change` を即バインドしており、DPG が初期描画の `default_value` をそのまま callback に渡すため `ParameterStore.set_override()` が呼ばれてしまうことにある。
- 目標は「すべてのパラメータを GUI に表示しつつ、ユーザーが操作するまではスクリプト側の値（例: `t` 連動）がそのまま流れる」状態を保つこと。

## 方針
1. **UI 構築/同期時のガード追加**: `ParameterWindowContentBuilder` に「現在はプログラム的な更新中である」というフラグ/コンテキスト（`_suspend_callbacks`）を導入し、`mount_descriptors` や `sync_*` 実行中に発火する callback では `ParameterStore.set_override` をスキップする。
2. **callback 入口の網羅**: `_on_widget_change` だけでなく `store_rgb01`（カラー系）や `_on_cc_binding_change`、vector 成分更新など override を行うすべての経路でガードを参照する。
3. **安全な順序**: UI 構築 → guard 開放 → Store からの初期同期、という順で処理し、同期中は既存の `_syncing` も併用してループを防ぐ。
4. **検証とテスト**: Parameter GUI なし/あり両方で `ParameterRuntime` の値更新が継続することを自動テスト化し、`sketch/251203.py` でも実機確認する。

## 作業チェックリスト
- [ ] **再現と原因特定**: `sketch/251203.py` を Parameter GUI 有効で起動し、`effect@p0.displace#2.t_sec` の `ParameterStore` エントリに GUI 初期化段階で `override` が入ることをログかデバッガで確認。DPG が初期 `default_value` で callback を叩く事実を把握する。
- [x] **ガード実装**: `src/engine/ui/parameters/dpg_window_content.py` に UI 構築/同期用のガード（例: `self._suspend_callbacks(count=0)`）を追加し、`mount_descriptors`・`sync_style_from_store`・`sync_palette_from_store` などの呼び出しをガード下に置く。`_on_widget_change`・`store_rgb01`・`_on_cc_binding_change` など override 系 callback 入口ではガード中なら早期 return する。
- [x] **副作用確認**: CC バインドや min/max 入力など他のウィジェットでも初期化時に override されないことを確認し、必要なら共通ヘルパへ置き換えて重複実装を整理する。
- [x] **回帰テスト**: DPG API をスタブ化したユニットテスト（`tests/ui/parameters` 配下）を追加し、「ウィジェット生成時に callback が呼ばれても ParameterStore の `override` は設定されない」ことを検証。加えて `ParameterRuntime` + `E.displace(t_sec=t * 0.1)` を 2 フレーム解決し、GUI 未操作時に `t_sec` が `t` に追従するテストを追加する。
- [x] **ドキュメント更新**: `docs/spec/parameter_gui_show_all_params_plan.md` へ今回のガード設計を追記し、「GUI 操作が無ければ実装側の値が優先される」旨とテスト方針を明記する。
- [ ] **動作確認**: `sketch/251203.py` と他の時間依存エフェクト（例: `E.wobble`）で Parameter GUI を開いたまま `t` モジュレーションと GUI override が両立することを手動確認し、必要ならスクリーンショット/ログを保存する。
