# dpg_window_content エラー処理・後方互換ロジック改善計画

目的: `src/engine/ui/parameters/dpg_window_content.py` に散在しているエラー処理や後方互換ロジックを整理し、挙動を変えずに見通しを良くする。特に、カテゴリ/グルーピング・DPG テーブル/テーマ・Display/HUD config 周りを優先的に単純化する。

## 対象ロジックと優先度

1. カテゴリ/グルーピング周りの後方互換（優先度: 高）
2. DPG テーブル policy/テーマ適用のフェイルセーフ乱立（優先度: 高）
3. Display/HUD config 読み込みとデフォルト処理（優先度: 中）
4. Store 同期まわりのフォールバック（優先度: 中）
5. CC 入力パースとバインディング（優先度: 低・挙動維持）
6. カラー正規化まわり（優先度: 低・挙動維持）

以下では、それぞれについて「現状 → 目標 → 具体タスク」を整理する。

---

## 1. カテゴリ/グルーピング後方互換の整理（優先度: 高）

### 現状

- `_build_grouped_table` / `_category_kind` で
  - `category_kind` が無い Descriptor に対して `source`（"shape"/"effect"）から `"shape"`/`"pipeline"` を推定している。
  - グルーピング key は `(kind, category)` だが、kind の決定ロジックが try/except ベースで分散している。

### 目標

- Descriptor 生成側（`state.py` / `value_resolver.py`）で `category_kind` を必ずセットする前提を強め、Content 側のフォールバックを最小限にする。
- `_build_grouped_table` は「(kind, category) でグループ化する」というロジックだけを担当し、`kind` の決定は `_category_kind` または Descriptor 自体に委ねる。

### タスク

- [x] `ParameterDescriptor` 生成経路を確認し、`category_kind` が常に設定されていることを検証する（`state.py` / `value_resolver.py` / `manager.py`）。
- [x] `_build_grouped_table` 内の
  - `try: kind = d.category_kind except: kind = "pipeline" if d.source == "effect" else "shape"`  
  を削減し、`kind = d.category_kind` 前提に寄せる。
- [x] `_category_kind` の `source` フォールバックも、実際に必要なケースが残っているかを確認し、不要なら削除またはログみに縮小する。
- [ ] 上記変更後、代表的なスケッチ（shape/effect 混在・HUD 利用）のカテゴリヘッダ表示が従来と変わらないことを目視確認する。

---

## 2. DPG テーブル policy/テーマ適用フェイルセーフの整理（優先度: 高）

### 現状

- `_build_grouped_table` / `build_display_controls` などで
  - `_dpg_policy([...])` の戻り値をそのまま `policy` に渡し（現在は int 化で吸収）、`None` の可能性を含んだまま扱っている。
  - `get_category_header_theme` / `get_category_table_theme` / `bind_item_theme` を個々の呼び出しで `try/except` し、「テーマ適用に失敗しても続行する」ロジックが散在している。

### 目標

- DPG バージョン差異や theme 生成の失敗は `ParameterWindowThemeManager` 側でできるだけ吸収し、Content 側は
  - `policy` はすでに安全な `int` として受け取る
  - theme は `None` なら何もしない、程度の分岐にとどめる
 という形に単純化する。

### タスク

- [x] `_dpg_policy` を ThemeManager に移すか、少なくとも「返り値は常に int」を保証する API に変更する（`None` フォールバックはそこで吸収）。
- [x] `_build_grouped_table` / `build_display_controls` で `policy` を受け取る位置を、「int 受け取り」に統一する（`_dpg_policy` 呼び出しを上位に寄せる）。
- [x] テーマ取得・バインド周りの `try/except` を見直し、ThemeManager 側で例外を握る方針に揃える。
  - Content 側は `th = theme_mgr.get_category_header_theme(...)` が `None` かどうかだけを見る形にする。
- [ ] 代表的な parameter_gui スキッチで、カテゴリテーマ/行背景テーマの適用が従来と変わらないことを確認する。

---

## 3. Display/HUD config 読み込みとデフォルト処理（優先度: 中）

### 現状

- `build_display_controls` で
  - `load_config` を try/except で呼び、失敗時に空 dict を返す lambda をセット。
  - `cfg`/`canvas`/HUD 設定に対し、`isinstance(..., dict)` をその都度チェックしている。
  - Store の値が無い場合や正規化失敗時は `_safe_norm` + default 値でフェイルソフトしている。

### 目標

- config 読み込みと型正規化は別ヘルパ関数（または util）に寄せ、Content 側では「正規化済みの構造」を信じる割合を増やす。
- Display/HUD のロジックを読むときに、「設定が壊れている場合の防御コード」をなるべく後ろに押しやる。

### タスク

- [x] `util.utils.load_config` の戻り値が壊れている場合にどうなりうるかを洗い出す。
- [x] `build_display_controls` 内の config 取得を、小さなヘルパ（例: `_resolve_canvas_colors()` / `_resolve_hud_defaults()`）に切り出す。
  - このヘルパ内で dict チェックや default 値適用を完結させる。
- [x] Content 側の本体ロジックは「色/初期値がすでに正しいタプルで渡される」という前提で書き直し、`isinstance(..., dict)` チェックを減らす。
- [ ] `parameter_gui` 関連の config (`configs/default.yaml` など) を読み込んだ上で、Display/HUD の初期状態が従来と同じになることを確認する。

---

## 4. Store 同期まわりのフォールバック（優先度: 中）

### 現状

- `on_store_change` で
  - Display/HUD ID は `_safe_norm` ベースでフェイルソフトしつつ DPG に反映。
  - その他 ID が DPG アイテムに存在しない場合に `get_descriptor` を試み、vector のみ特別扱いする。
  - vector の場合でも「値が list/tuple でない」「対応するアイテムが無い」状況を if/continue でスキップしている。

### 目標

- Store 側が「登録済み ID だけ通知する」がより強く保証されていれば、Content 側での `KeyError` ガードや「型が変なときも黙ってスキップ」をある程度減らせる。
- ただし、バグ由来の通知で UI 全体が落ちるのは避けたいので、「ログを出して戻る」程度のフェイルセーフは維持する。

### タスク

- [x] ParameterStore の `_notify` と `subscribe` 利用箇所を再点検し、「登録済みでない ID が通知されうるパス」があるか確認する。
- [x] `on_store_change` 内で `get_descriptor` を呼ぶパターンを洗い出し、実際に KeyError が起こり得るかどうかをテスト/ロジックで確認する。
- [x] ありえないパスであれば、`try/except` を削るのではなく「明示的に assert/ログ」に変えることで、挙動は保ちつつコード意図を明確化する。
- [ ] vector 更新ロジックも、値型前提（list/tuple）を上位で保証できないか検討する（必要なら ParameterStore 側で型チェック）。

---

## 5. CC 入力パースとバインディング（優先度: 低）

### 現状

- `_on_cc_binding_change`:
  - 入力が空ならバインディング解除。
  - `int(float(text))` パース失敗時は解除＋ UI を空に戻す。
  - 値を 0..127 にクランプ。
- `_add_cc_binding_input*`:
  - `cc_binding` 取得を try/except で包み、問題があれば `None` として扱う。

### 目標

- 挙動（入力が多少おかしくても落とさず、合理的に扱う）はそのまま維持する。
- 実装は「どのように防御しているか」が読みやすい形に整理する（共通化・コメント）。

### タスク

- [x] `_on_cc_binding_change` のロジックに短い docstring/コメントを付与し、例外パターンごとの扱いを明文化する。
- [x] `_add_cc_binding_input` / `_add_cc_binding_input_component` の重複ロジック（`default_text` 決定）をヘルパにまとめ、行数を減らす。
- [x] CC バインディングに関するユニットテスト（既存の有無を確認し、なければ最小限のもの）を検討する。

---

## 6. カラー正規化まわり（優先度: 低）

### 現状

- `_safe_norm` は `normalize_color` 失敗時に default を返す。
- `store_rgb01` は `normalize_color` 全般の例外を握り、`logger.exception` 出力後に Store 更新をスキップ。
- style.color（vec3）と RGBA（vec4）の分岐もここに含まれる。

### 目標

- 挙動（色指定が壊れていても落とさず、既定値か何も更新しない）はそのまま維持する。
- 仕様として「style.color は vec3 扱い、その他は RGBA」といったルールを docstring とコメントで明文化する。

### タスク

- [x] `_safe_norm` / `store_rgb01` の docstring に、入力/出力の仕様と失敗時の扱いを明記する。
- [x] style.color の特別扱いルール（vec3 保存）の理由を短くコメントとして残す（`style.color は HUD と同じ 0–255 表示に統一` など）。
- [x] 必要であれば architecture.md の parameter_gui セクションに「カラー正規化の仕様」として一言追記する。

---

この計画では、まずカテゴリ/グルーピングと DPG テーブル/テーマ周りから整理を始め、それ以外の防御コードについては様子を見ながら段階的にスリム化していく方針とする。実際に着手する際は、各ステップごとに pyright/mypy/ruff と簡単な動作確認を挟みながら進める。 
