# src/engine/ui/parameters/value_resolver.py レビュー & 改善計画

## 1. 現状レビュー

### 1-1. 良い点

- モジュール先頭の日本語ドキュメントで責務とデータフローが簡潔に説明されており、役割の理解がしやすい。
- `ParameterContext` により、スコープ・名前・インデックスなど呼び出しコンテキストが一箇所に集約されている。
- `resolve()` が scalar/vector/passthrough の3系統に明示的に分岐しており、値種別ごとの扱いが追いやすい。
- RangeHint / VectorRangeHint を用いた UI 向けメタ情報の組み立てが分離されており、数値レンジの扱いが整理されている。
- CC バインドによる上書き処理が `_register_scalar` / `_resolve_vector` に閉じており、呼び出し元からは「登録して実値を受け取る」インターフェースに見える。
- 型ヒントが全体的に付いており、値種別 (`ValueType`) を経由した判定など、暗黙の型変換が少なくなっている。

### 1-2. 気になった点・課題

- `category` / `category_kind` / `pipeline_uid` / `step_index` など、Descriptor 共通の情報組み立てロジックが `_resolve_scalar` / `_resolve_vector` / `_resolve_passthrough` の3か所に重複している。
- `resolve()` 内での `param_order_map` 構築がその場で `try/except` を含んだブロックとして書かれており、責務ごとの見通しがやや悪い。
- `_resolve_vector` は「デフォルト値の正規化」「Descriptor の登録」「CC バインドの適用」「フォールバック」の複数ステップを1メソッド内で担っており、読みながら頭のスタックが増えやすい。
- `_resolve_passthrough` も value_type 判定・choices 抽出・string 用メタ情報解釈・Descriptor 作成を一気に行っており、処理の意図をコメントで補っているが、関心ごとを分ける余地がある。
- `except Exception:` が広い範囲で多用されており、フェイルソフトな設計方針は理解できるものの、「どこまでを想定エラーとして握りつぶしているか」がコードから読み取りにくい。
- `_component_default_actual` が未使用になっており、過去の実装の名残が残っているように見える（削除候補）。
- モジュール定数 `_VECTOR_SUFFIX` も現在は使用されておらず、ローカルに同等のタプル `("x", "y", "z", "w")` が定義されている（どちらかに統一したい）。
- `_range_hint_from_meta` の引数 `value_type` / `default_value` が現状未使用で、インターフェースと実装のギャップがある。
- `_resolve_vector` の引数 `has_default` も現状未使用であり、将来拡張か実装漏れかが分かりにくい。
- `inspect._empty` を直接参照しており、型チェッカー抑制コメント（`type: ignore`）が散見されるため、Python 標準 API に合わせて素直に書き換える余地がある。

## 2. 実装改善アクションチェックリスト

- [x] `ParameterContext` に Descriptor 共通情報（`category` / `category_kind` / `pipeline_uid` / `step_index` 相当）を返すプロパティを追加し、`_resolve_scalar` / `_resolve_vector` / `_resolve_passthrough` から重複ロジックを排除する。
- [x] `resolve()` 内の `param_order_map` 構築処理を `_build_param_order_map(signature, skip)` のような小さなヘルパに切り出し、`try/except` 範囲を局所化して読みやすくする。
- [x] Descriptor 生成処理の共通部分（`id` / `label` / `source` / `category` / `category_kind` / `pipeline_uid` / `step_index` / `param_order` など）をヘルパ関数または小さなビルダにまとめ、各 `_resolve_*` メソッドの引数数と重複を減らす。
- [x] `_resolve_vector` を「デフォルト値の正規化」「Descriptor の登録とストア解決」「CC バインド適用」の3ステップに論理的に分割し、少なくとも CC 周りの処理を `_apply_cc_to_vector(descriptor_id, base_values)` のようなヘルパに切り出す。
- [x] `_resolve_passthrough` の中で行っている value_type 判定・choices 抽出・string 用メタ情報解釈を、それぞれ小さな private ヘルパ（例: `_extract_choices(meta)` / `_string_meta(meta)`）に分け、メインのロジックを「Descriptor を作る」ことに集中させる。
- [x] 未使用のコード・引数を整理する（`_component_default_actual` / `_resolve_vector` の `has_default` 引数 など）。削除時に呼び出し元のシグネチャも合わせて簡素化する。
- [x] `inspect._empty` 直接参照を `inspect.Parameter.empty` に統一し、`# type: ignore` を削減して型情報と実装を一致させる。
- [ ] `_determine_value_type` / `_is_vector_value` / `_range_hint_from_meta` など meta ベースの判定周りに、仕様意図を短く補足するコメントを追加し、RangeHint と GUI 表示の関係をコードから読み取りやすくする。
- [ ] 仕様に影響しやすい箇所（vector の次元決定ロジック、CC バインド適用、enum/string の supported 判定）について、既存テストの有無を確認し、必要に応じて最小限のテストを追加してリファクタリングの安全性を高める。

## 3. 次のステップ

- 上記チェックリストの内容・粒度で問題ないか確認してほしい。
- 問題なければ、この md のチェックボックスを更新しながら上から順に実装を進める（完了した項目は `[x]` に変更する）。
- 実装中に追加で気づいた点や、事前に相談したい設計上の論点が出てきた場合は、この md ファイルに追記してから報告する。
- 現時点ではコードへの変更は一切行っていない。
