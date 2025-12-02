# Parameter GUI 全パラメータ表示・GUI優先 改善計画

## 背景と課題
- `ParameterValueResolver` はシグネチャ既定値のみ ParameterStore に登録し、明示指定値は素通りさせるため GUI に出ない。
- GUI 変更で override したくても Descriptor が無く制御が移らない。explicit 指定時の非表示ロジックをなくせば挙動統一とコード削減が見込める。

## 進め方（チェックリスト）
- [ ] 現行挙動の把握: resolve() の default/provided 分岐と scalar/vector/passthrough それぞれの register/resolve フロー、snapshot/persistence への影響を整理する。
- [ ] 仕様決定: すべてのパラメータを Descriptor 登録し、初期 original に provided を反映しつつ default_value はシグネチャ由来を保持する方針を固める。優先順位（GUI > 明示値 > 既定値）と skip 対象（例: g）の扱いを確認する。
- [ ] 実装: ValueResolver の provided 分岐を廃止し、register/resolve を共通化する（scalar/vector/passthrough）。必要なら ParameterStore の original 更新と override 維持ロジックを調整して GUI 変更が確実に実行値へ反映されるようにする。
- [ ] 表示/永続化の整合: store.descriptors() 起点の DPG レイアウトが全パラメータを拾うことを確認し、persistence/snapshot が新規 Descriptor を保存・復元できるかを点検する。不要になった「表示しない」関連コードを削除する。
- [ ] ドキュメント整備: 変更後の優先順位と登録ポリシーを architecture.md や該当モジュールのヘッダコメントに反映する。
- [ ] テスト計画: 追加/更新するテストを決める（例: 明示指定パラメータが GUI に表示され override が実値を置き換えるシナリオ、ベクトル/enum を含む）。変更ファイルに対して `ruff`/`black`/`isort`/`mypy` と関連 `pytest -q`（`tests/ui/parameters` など）を実行する。
