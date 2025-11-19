# parameter_gui poly_effect ラベル重複対応 改善計画

目的:  
`sketch/251118.py` のように同一ラベルの `E`（例: `"poly_effect"`）が複数回呼ばれた場合、Parameter GUI 上で:
- パイプライン毎のヘッダ（カテゴリ）が `poly_effect_1`, `poly_effect_2`, ... のように自動連番付きで分かれて表示されること
- どのパラメータ行がどのヘッダ（パイプラインインスタンス）に属しているかが一目で分かること
を実現する。

## 前提・現状理解

- Parameter GUI は `ParameterDescriptor`（`src/engine/ui/parameters/state.py`）と `ParameterWindowContentBuilder`（`src/engine/ui/parameters/dpg_window_content.py`）でメタ情報とレイアウトを管理している。
- エフェクト由来パラメータは `ValueResolver`（`src/engine/ui/parameters/value_resolver.py`）で `ParameterDescriptor` に変換される。
- 現状 `ParameterDescriptor.category` は:
  - shape: 形状名
  - effect: `context.pipeline_label or context.pipeline or context.scope`
 となっており、同じパイプラインラベル（例: `"poly_effect"`）のエフェクトが複数あっても区別されない。
- `ParameterDescriptor.label` は `f"{context.label_prefix}: {param_name}"`（実体は `effect_name#index: param_name`）の形式で、すでにエフェクトインスタンス番号（`index`）を含んでいる。
- `E.label("poly_effect")` で指定したラベル文字列は `pipeline_label` として `ParameterContext` に渡され、最終的に `ParameterDescriptor.category`（＝GUI のカテゴリヘッダ文字列）として使われる。

## 改善方針（全体像）

- エフェクトコンテキストごとに「インスタンス番号」を付与し、Parameter GUI 用のカテゴリ名（ヘッダ表示）に反映する。
- 具体的には:
  - `poly_effect` というラベルを持つエフェクトが N 回登場した場合、
    - カテゴリ: `poly_effect #1`, `poly_effect #2`, ... のように分割されたヘッダとして表示されること
    - すでに行ラベル（`effect_name#index: param_name`）にはインスタンス番号が含まれているため、まずはカテゴリヘッダ側の分離にフォーカスする。
- ID の一意性は既に担保されているため、既存の保存データとの互換性をなるべく維持しつつ、「カテゴリ名（ヘッダ）のみ」を変える方向を基本とする。

## 実装タスクチェックリスト

- [ ] 現状のエフェクトコンテキスト情報の構造を確認する  
  - [ ] `ParameterContext`（定義箇所: `src/engine/ui/parameters/value_resolver.py` 付近もしくは関連モジュール）で、同一 `pipeline_label` / `pipeline` を持つエフェクトがどのように識別されているか調査する  
  - [ ] `context.index` や内部 UID など、インスタンス番号候補となるフィールドの有無を確認する

- [ ] エフェクトインスタンス番号の定義と算出方法を決める  
  - [ ] 「同じ `pipeline_label`（または `pipeline`）を持つエフェクトを 1, 2, 3, ... と連番付けする」というルールを明文化する  
  - [ ] ランタイム内でエフェクト列を走査してインスタンス番号を振るか、`ParameterContext` 生成時点でインスタンス番号を含めるかを決定する  
  - [ ] 既存の `step_index`（パイプライン内の順序）との違いを整理し、役割の衝突が起きないようにする

- [ ] `ParameterDescriptor` への情報反映方針を決める  
  - [ ] インスタンス番号を `ParameterDescriptor` に伝えるためのフィールド（例: `instance_index`）を追加するか検討する  
  - [ ] もしくは、`category` と `label` を生成するタイミングでインスタンス番号を埋め込むだけにして、新フィールド追加を避けるかを比較検討する  
  - [ ] `category_kind`（`\"pipeline\"` など）と `parameter_gui.theme.categories` の対応に影響がないように設計する

- [ ] カテゴリ名生成ロジックの更新案を設計する  
  - [ ] 現状のカテゴリ決定ロジック（`ValueResolver._resolve_scalar/_resolve_vector` 内の `category` 変数）を整理する  
  - [ ] エフェクト由来の場合、`category = f\"{context.pipeline_label} #{instance_index}\"` のように連番付きカテゴリを生成する案を具体化する  
  - [ ] `parameter_gui.theme.categories` でカテゴリ単位のテーマ設定を行っている場合の影響を整理し、必要であれば「ベースカテゴリ名（例: `poly_effect`）とインスタンス番号付きカテゴリ名の両方」を扱える設計を検討する

- [ ] パラメータラベル生成ロジックの更新案を設計する  
  - [ ] 現状の `label=f\"{context.label_prefix}: {param_name}\"`（＝`effect_name#index: param_name`）がどの程度分かりやすいかを評価する  
  - [ ] 本タスクではパラメータ行ラベルの仕様変更は必須とせず、必要であれば別タスクとして「行ラベルの簡略化（例: `scale`, `offset` のみ）」を検討する

- [ ] GUI レイアウト層でのカテゴリ分割を確認する  
  - [ ] `ParameterWindowContentBuilder`（`src/engine/ui/parameters/dpg_window_content.py`）の `mount_descriptors` / `_build_categories` 周辺を確認し、`category` 単位でテーブル/ヘッダが作られていることを再確認する  
  - [ ] カテゴリ名が `poly_effect #1`, `poly_effect #2` のように変わったときに、想定通り別ヘッダとして扱われるかを確認する  
  - [ ] HUD/Display カテゴリなど他の `category_kind` に影響が出ないかを軽くチェックする

- [ ] 永続化・互換性の影響を洗い出す  
  - [ ] `src/engine/ui/parameters/persistence.py` での `ParameterDescriptor` シリアライズ形式を確認し、`category`・`label` の変更が保存データの読み込みに与える影響を整理する  
  - [ ] 既存の GUI 状態ファイル（`data/gui/*.json`）のキーが `id` ベースかどうかを確認し、ラベル変更のみであれば互換性を維持できるかを判断する  
  - [ ] 互換性に問題が出る場合の移行方針（例: 古いカテゴリ名を受け入れるフォールバック）を検討する

- [ ] 実装順序のドラフトを作る  
  - [ ] まず `ParameterContext` まわりにインスタンス番号の概念を追加し、それを `ValueResolver` に渡す  
  - [ ] 次に `ValueResolver` で `category` / `label` の生成ロジックを更新し、インスタンス番号を反映する  
  - [ ] 必要に応じて `ParameterDescriptor` のフィールドや `ParameterWindowContentBuilder` のカテゴリ処理を最小限修正する  
  - [ ] 最後に `sketch/251118.py` を使って実機確認し、GUI 上で `poly_effect_1`, `poly_effect_2`, ... のようにヘッダとラベルが分かれていることを確認する

- [ ] テストと検証方法を整理する  
  - [ ] 単体テストで `ParameterDescriptor` の `category` / `label` が期待通りになることを検証するテストケース追加を検討する  
  - [ ] 実行確認として `python main.py sketch/251118.py` 相当のコマンドで Parameter GUI を起動し、視覚的にヘッダ分割とラベル連番を確認する手順をまとめる  
  - [ ] 他の代表的なエフェクトスケッチで regression がないか軽くチェックする

## メモ・確認事項（ユーザー確認したい点）

- [ ] カテゴリ名のフォーマット: `poly_effect #1` のように `#` 区切りでよいか、それとも `poly_effect (1)` や `poly_effect[1]` のような表記がよいか  
- [ ] 既存の GUI 状態ファイルとの互換性よりも、分かりやすい表示を優先してよいか（互換性が少し崩れても問題ないか）
