# CC と GUI の同期/優先度 改善計画（チェックリスト）

目的
- GUI で値を変更後でも、次の CC 入力でパラメータと GUI 表示が CC 値へジャンプし、その後も CC で継続制御できる状態にする。

背景（原因の要約）
- 現状 `draw()` 内で `cc[idx]` の数値をそのままエフェクト/シェイプ引数へ渡している箇所がある（例: `scale=(1,1,cc[2])`）。
- 数値直渡しは「MIDI 由来」であることを識別できず、GUI で override された後は `original` 更新として扱われて CC が優先されないため、以後 CC 変化が反映されにくい。
- CCBinding（`CC(idx, map=...)`）経路なら `midi_override` がセットされ、優先度 `midi > gui > original` が効くため期待挙動になる。

方針（最小変更）
- CC で駆動したい引数は CCBinding を用いて渡す（数値直渡しをやめる）。
- 既存の優先順位（`midi_override > override > original`）を前提に、Resolver/GUI 通知の既存実装を活用する。
- 変更はサンプル（`main2.py` 他）とドキュメント、最小の単体テストを追加。

影響範囲
- サンプル `main2.py`（必要なら `main.py`）
- ドキュメント（`architecture.md` のコード例）
- テスト（`tests/ui/parameters` 配下に最小ケース追加）

完了条件（DoD）
- `main2.py` を起動し、
  1) CC で動く → 2) GUI で値変更 → 3) CC を再操作 で GUI/値が CC にジャンプし、以後 CC で追従することを目視確認。
- 追加テストが緑（`pytest -q tests/ui/parameters/test_cc_binding_priority.py`）。
- `ruff/mypy` が変更ファイルに対して緑。
- ドキュメント更新（`architecture.md` の例が CCBinding を示す）。

作業チェックリスト

0) 現状確認（実サンプルでの CCBinding 使用有無）
- [x] リポジトリで `CC(` の使用箇所を検索し、サンプルで未使用であることを確認（docs/や cc_binding.py 自身を除く）。
- [x] `main2.py` が `cc[2]` の数値直渡しであることを確認（`main2.py:16` 近辺）。
- [x] `main.py` でも CCBinding 未使用であることを確認（必要に応じて対象外とする）。

1) 実装の前提確認（優先度/通知）
- [ ] 優先度が `midi > gui > original` であることを確認（src/engine/ui/parameters/state.py:39 付近の `ParameterValue.resolve`）。
- [ ] GUI 側が store 通知で値更新することを確認（src/engine/ui/parameters/dpg_window.py:265 の `_on_store_change`）。

2) サンプル（main2）の CCBinding 化
- [x] `main2.py` に `from engine.ui.parameters.cc_binding import CC` を追加（main2.py:1 付近）。
- [x] `affine(scale=(1, 1, cc[2]))` を `affine(scale=(1, 1, CC(2)))` へ変更（main2.py:14 付近）。
- [ ] 必要に応じてレンジに合わせた map を付与（例: `CC(2, map=lambda v: v)` または `CC(2, map=lambda v: 0.25 + 3.75*v)`）。
- [x] （任意）`G.polyhedron().scale(200 * cc[1])` は GUI 対象外のため現状維持で OK（Geometry メソッドはランタイム対象外）。

3) メインサンプルの整合（任意）
- [ ] `main.py` でも CC 駆動したい effect/shape 引数があれば CCBinding を使用（main.py:16 付近）。

4) テスト追加（最小）
- [x] 新規 `tests/ui/parameters/test_cc_binding_vector.py` を追加（ベクトル内 CCBinding が GUI より優先されること）。
- [x] 実行: `pytest -q tests/ui/parameters/test_cc_binding_vector.py`。

5) ドキュメント更新
- [ ] `architecture.md:110` 付近のサンプルを CCBinding 例へ更新（`rotate(...= (CC(3), CC(4), CC(5)))` など）。
- [ ] README から該当箇所への誘導が必要なら追記。

6) 動作確認（手動）
- [ ] 起動: `python main2.py`。
- [ ] CC で対象パラメータが動くことを確認。
- [ ] GUI で当該スライダーを変更し値が反映されることを確認。
- [ ] その後 CC を再操作し、GUI/値が CC にジャンプして以後追従することを確認（意図どおり“値飛び”可）。

7) 仕上げ
- [ ] 変更ファイルに限定して Lint/Format/Type: `ruff check --fix {changed}`, `black {changed} && isort {changed}`, `mypy {changed}`。
- [ ] 変更差分の要約と根拠（このチェックリスト + 動作確認メモ）を PR に記載。

備考/オプション（今回は見送り）
- [ ] `CC(index, scale=..., offset=...)` の糖衣やカーブ/デッドゾーンは将来検討（複雑化回避のため今回は採用しない）。
- [ ] `ParameterWindowController.apply_overrides()` で CC→store へのブリッジは行わない（CCBinding 経路で十分かつ単純）。

参照
- main2 サンプル: main2.py:14
- 優先度: src/engine/ui/parameters/state.py:39
- GUI 反映: src/engine/ui/parameters/dpg_window.py:265
- CCBinding: src/engine/ui/parameters/cc_binding.py:1
- Resolver（CCBinding ハンドリング）: src/engine/ui/parameters/value_resolver.py:118, 292
  - ベクトル経路の改善: src/engine/ui/parameters/value_resolver.py:193-286

---

この計画で問題なければ指示ください。承認後、上記チェックを順に実施します。
