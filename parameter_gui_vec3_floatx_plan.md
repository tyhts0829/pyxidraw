# Parameter GUI: Vec3 を floatx×3 スライダで扱う 改善計画

目的: パラメータ GUI において、Vec3（例: `angles_rad: Vec3`）の未指定引数に対し、float スライダを 3 本（x/y/z）並列表示で操作できるようにする。既存の `ParameterStore`/`ValueResolver` の挙動（成分ごとの Descriptor 登録・`vector_group` によるグルーピング情報付与）を活かし、UI 側で見た目を一体化する。

## 現状と課題
- 既存の解決: `engine.ui.parameters.value_resolver.ParameterValueResolver` は、Vec3 既定値採用時に各成分を `value_type="float"` の Descriptor として登録し、`vector_group` に元のパラメータ ID を設定している（例: `effect.rotate#1.angles_rad.x`）。参照: `src/engine/ui/parameters/value_resolver.py:190` 以降の `_resolve_vector()`。
- UI 側: `src/engine/ui/parameters/dpg_window.py:159` の `mount()` は Descriptor を 1 つずつ行にして描画し、`_create_widget()` は `value_type="float"` を単一スライダとして扱う。`value_type="vector"` 分岐はあるが、Resolver は「ベクトルの親 Descriptor」を発行していないため未使用。
- ギャップ: Vec3 を 3 本の float スライダとして“1 行にまとめて”表示・編集できる仕組みがない（各成分がバラバラの行になる）。

## 目標（要件）
- Vec3 の各成分（x/y/z）を 1 行に横並びで表示し、float スライダで編集できること（以降「floatx×3」）。
- ラベルは親パラメータ名（例: `rotate#1 · angles_rad`）を左列に 1 つだけ表示。右列に x/y/z の 3 スライダを配置。
- ストアの ID は従来通り各成分 ID（例: `...angles_rad.x`）をタグとして用い、双方向同期（GUI → Store → GUI）が保たれること。
- RangeHint（min/max/step）が成分ごとにあれば尊重。無い場合は `ParameterLayoutConfig.derive_range()` の既定（0..1）を使う。
- 優先順位ポリシーは維持（明示引数 > GUI > 既定値）。提供値がある場合は GUI に登録しない（現仕様のまま）。

## 実装方針（最小差分）
1) グルーピング導入（UI 側のみ）
   - `dpg_window.ParameterWindow.mount()` 内で、`vector_group`（非 None）をキーに Descriptor をまとめる。
   - グループに属さない Descriptor は従来通り 1 行で描画。
   - グループに属する Descriptor は「1 行＝親パラメータ」「右列＝x/y/z 3 スライダ」を生成。

2) 成分順序・表示
   - サフィックス `.x/.y/.z(.w)` を優先順に整列（`x,y,z,w`）。
   - ラベルは左列に親名（`{name}#{index} · {param}`）。右側に各スライダのミニラベル（x/y/z）またはツールチップ無しのアイテム間余白で区別。

3) スライダ生成
   - 既存の数値スライダ生成ロジックを再利用（`_create_widget()` の float/int 分岐）。ただしグループ行では 3 つの `dpg.add_slider_float` を横並びで追加。
   - `tag` は各成分の `descriptor.id` をそのまま使用（Store の通知と一致）。
   - 値/レンジは各成分の `range_hint` を優先し、無ければ `ParameterLayoutConfig.derive_range()` を使用。

4) 同期処理
   - `_on_widget_change()` は既存のままで、各成分スライダの `user_data` に成分 ID を渡す。
   - `_on_store_change()` も既存のままで、各成分 ID に対して `dpg.set_value()` を行う。

5) 後方互換・削除しないもの
   - `value_type=="vector"` 分岐は残す（将来の親 Descriptor 対応に備えたフォールバック）。ただし現仕様では利用されない。

## 変更ファイル（想定・参照）
- `src/engine/ui/parameters/dpg_window.py:159` 付近（`mount()` 内の並べ替えと行生成）
- `src/engine/ui/parameters/dpg_window.py:201` 付近（`_create_row()` からベクトル行に委譲する処理を追加）
- 必要ならヘルパ（同ファイル内関数）を追加: `__group_descriptors_by_vector_group(items)` と `__create_vector_row(parent, group_key, items)`

## テスト計画（編集ファイル優先の高速ループ）
- 既存テストの活用
  - `tests/ui/parameters/test_value_resolver.py::test_parameter_value_resolver_handles_vector_params_defaults_register_gui` は、成分 ID が登録されることを確認済み（UI 変更不要）。
- 追加（純ロジックのユニットテスト）
  - `dpg_window.py` 内に「グルーピングだけ」を返す純関数（副作用なし）を切り出し、`tests/ui/parameters/test_layout_grouping.py` を新設。
    - 入力: `ParameterDescriptor` 群
    - 期待: `vector_group` ごとに `x,y,z` 整列・その他は単独行
- DPG 依存のマウント E2E は最小（`test_dpg_mount_smoke.py` の範囲）に留める。

実行コマンド（変更ファイル限定）:
- Lint: `ruff check --fix src/engine/ui/parameters/dpg_window.py`
- Format: `black src/engine/ui/parameters/dpg_window.py && isort src/engine/ui/parameters/dpg_window.py`
- TypeCheck: `mypy src/engine/ui/parameters/dpg_window.py`
- Test (UI/parameters): `pytest -q tests/ui/parameters -k grouping`

## 具体的作業チェックリスト
- [ ] dpg_window: グループ化ロジックを実装（`vector_group` 単位）
- [ ] dpg_window: Vec3 行のスライダ 3 本生成（x/y/z 並列）
- [ ] dpg_window: `_create_row()` をベクトル行と単独行で分岐
- [ ] dpg_window: `mount()` でカテゴリ内の項目をグループ化して描画
- [ ] テスト: グルーピング純関数を追加しユニットテスト
- [ ] ruff/black/isort/mypy（変更ファイルのみ）を通す
- [ ] `pytest -q tests/ui/parameters` 緑（smoke + 追加テスト）
- [ ] `architecture.md` に「Vec3 は GUI で floatx×3 スライダをグループ表示」追記（参照先コードも明記）

## 確認したいこと（要回答）
- 表示形式:
  - 右列の 3 スライダは「水平に並べる」で良いか。各スライダのミニラベル（x/y/z）を表示するか、省略して並び順で判断するか。
  - スライダ幅の配分（等幅で 3 分割で良いか）。
- ステップ処理:
  - `RangeHint.step` がある場合、UI で丸める（量子化）か。現状は「クランプのみ UI、量子化は署名生成のみ」だが、UI 丸めを許容するか。
- Vec4 対応:
  - 将来の Vec4（w あり）も同実装で水平 4 本とするか（並び順 `x,y,z,w`）。
- ラベル表記:
  - 左列ラベルは `"{name}#{index} · {param}"` 固定で良いか。現在の成分行ラベル（`... · angles_rad.x`）からの変更に問題ないか。

## 互換性・影響
- API/Store 互換: 破壊的変更なし（ID/値解決は従来通り）。UI の見た目のみ改善。
- 既存の `value_type=="vector"` ウィジェット分岐は温存（未使用のまま）。

## 実施後の完了条件
- 変更ファイルに対して `ruff/black/isort/mypy` が成功。
- `pytest -q tests/ui/parameters` 緑。
- `architecture.md` と実装の整合（該当ファイル参照付きで更新）。

---

作業開始可否・上記の確認事項の方針をご指示ください。合意後、チェックリストを消し込みながら実装します。

