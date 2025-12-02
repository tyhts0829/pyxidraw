# Parameter GUI MIDI レンジ連動 改善計画

## 背景と課題
- Parameter GUI の min/max 入力は `ParameterStore.set_range_override()` に保存され、UI スライダーには反映されるが、MIDI の CC スケーリングは `param_meta` 由来のレンジ固定のまま。
- スカラは `value_resolver._register_scalar()`、ベクトルは `_apply_cc_to_vector()` / `snapshot.extract_overrides()` で `range_hint`/`vector_hint` を使っており、UI レンジ変更が MIDI に届かない。
- UI で設定したレンジを CC 適用にも使い、GUI と MIDI の挙動を一致させたい。

## 進め方（チェックリスト）
- [ ] 現行フロー整理: CC 適用箇所（`value_resolver` と `snapshot.extract_overrides`）と UI レンジ計算（`dpg_window_content._effective_range`、`ParameterStore`）のデータ流れを図示して、実効レンジの決定ポイントを明確化する。
- [ ] レンジ計算ヘルパー設計/実装: Descriptor の `range_hint`/`vector_hint` と `range_override` を合成して実効レンジを返す関数を追加し、異常値時のフォールバック（メタレンジ→既定 0..1）の方針を決める。
- [ ] CC スケーリング更新（スカラ）: `_register_scalar` で CC 適用時に新ヘルパーを参照し、UI の min/max 変更が実行値に反映されるようにする。
- [ ] CC スケーリング更新（ベクトル）: `_apply_cc_to_vector` と `snapshot.extract_overrides` の CC 適用で実効レンジを利用し、全成分に UI レンジ変更を共有する。
- [ ] UI 側のレンジ計算統合: `_effective_range` が新ヘルパーと重複しないよう整理し、UI と MIDI で同じ計算パスになることを確認する。
- [ ] 永続化の整合性確認: `persistence.save_overrides/load_overrides` の ranges が新ヘルパーでそのまま使えるか確認し、欠損時フォールバックのテストケースを洗い出す。
- [ ] テスト計画: 追加・修正するテストを決定（例: range override が CC スケーリングに効くスカラ/ベクトルの単体、永続化経路、`ParameterStore` の新ヘルパー）し、対象ファイルと期待結果を明記する。
- [ ] 検証: 変更ファイルを対象に `ruff`/`black`/`isort`/`mypy` と `pytest -q`（少なくとも `tests/ui/parameters` の該当ケース）を実行し、必要なら `architecture.md` の整合も確認する。
