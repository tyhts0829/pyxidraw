**目的**
- Parameter GUI の `font_index` を、実際に選択されるフォント（特に `.ttc` のフェイス数）に合わせて動的に 0..N に制限し、範囲外指定での実行時エラーを未然に防ぐ。

**現状と課題**
- `text` シェイプは `font_index` を公開（初期 0）。GUI 側のレンジはメタの既定値 0..32 に固定されている。
  - 参照: `src/shapes/text.py:402`（`__param_meta__` の `font_index: {min:0, max:32, step:1}`）
- 実フォントが `.ttc` の場合、ファイルが内包するフェイス数より大きな `font_index` を指定すると fontTools が例外を送出する。
  - 例: `TTLibFileIsCollectionError: specify a font number between 0 and 5 (inclusive)`
- GUI のヒント（0..32）が実フォントの実上限と同期していないため、UX として落とし穴が残っている。

**方針（案C）**
- 「メタの max 値を動的化」して GUI スライダの上限を、実際に解決されるフォントのフェイス数 N に合わせる。
- 初期登録時点（最初に `text(...)` が検出された瞬間）で正しいレンジ 0..N-1 を設定する。これにより通常利用での範囲外入力を防ぐ。
- 将来拡張で「フォントの切り替え（パラメータや設定の変更）にも追従して再構成」できる仕組みを用意（任意・後続）。

**実装概要**
1) フォントのフェイス数 N を取得するユーティリティを `TextRenderer` に追加
   - 新規: `TextRenderer.get_font_face_count(font_name: str) -> int`
   - 解決ロジックは `get_font()` と同一（パス優先 → 部分一致探索 → 既定フォールバック）。
   - `.ttc` は `fontTools.ttLib.TTCollection` から `len(collection.fonts)`、`.ttf/.otf` は 1。
   - `fontTools` 不在（ダミー環境）や取得失敗時は安全側で 1 を返す（＝ `font_index` は 0 のみ）。
   - 低コスト化のため、`_face_count_cache: dict[str, int]` を追加（キーは `font_name|resolved_path`）。

2) ParameterRuntime 側で `font_index` の RangeHint を上書き
   - `engine.ui.parameters.value_resolver.ParameterValueResolver.resolve()` で、`shape/text` の `font_index` に限り、`RangeHint(min=0, max=N-1, step=1)` を注入する。
   - 実装方法（最小改修）:
     - `resolve()` で構築した `merged`（デフォルトとユーザー指定をマージした辞書）を `_resolve_scalar()` に渡せるように引数を拡張する（デフォルト引数追加）。
     - `_resolve_scalar()` 内で `if context.scope=="shape" and context.name=="text" and param_name=="font_index":` なら `font_name = str(merged.get("font", "Helvetica"))` を取得し、`TextRenderer.get_font_face_count(font_name)` を呼んで RangeHint を生成。
     - それ以外のケースは従来通り `__param_meta__` のヒントを使用。

3) 既存メタの扱い
   - `src/shapes/text.py` の `__param_meta__` は、当面はフォールバック（未解決時の基本レンジ）として残す。
   - ただし実際の GUI 表示では 2) の動的ヒントが優先されるため、固定 0..32 は実害を生じない。

4) 将来拡張（任意）: ランタイム中の再構成
   - フォントの切替（`G.text(font=...)` をコード側で変更等）に追従するには、登録済み Descriptor の `range_hint` をアップデート→通知→DPG スライダーに `min/max` を反映する処理が必要。
   - 追加API案（後続PR）:
     - `ParameterStore.update_descriptor_range(id: str, hint: RangeHint) -> None` を追加し、更新時に `_notify({id})`。
     - `ParameterWindow._on_store_change()` に `if desc.value_type in {"int","float"} and dpg.does_item_exist(id): dpg.configure_item(id, min_value=..., max_value=...)` を追加（ベクタ成分は `::{x,y,z,w}` を同様に更新）。
   - 当面は「初回登録時に正しく設定する」だけで UX を大きく改善可能なため、本PRではスコープ外として設計のみ記載。

**変更範囲（最小）**
- 追加: `src/shapes/text.py`
  - `class TextRenderer` に `get_font_face_count(font_name: str) -> int` と `_face_count_cache` を追加。
  - 既存のファイル解決フロー（パス優先/部分一致/フォールバック）を再利用。
- 変更: `src/engine/ui/parameters/value_resolver.py`
  - `resolve()` → `_resolve_scalar()` の呼び出しを拡張して `merged_params` 参照を渡す。
  - `_resolve_scalar()` 内で `text.font_index` のときに `RangeHint(0, N-1, step=1)` を生成・適用。
- 変更なし: `src/engine/ui/parameters/dpg_window.py`
  - 初回登録時の RangeHint で適切にスライダが生成されるため、必須の変更はなし。
  - 将来拡張時のみ `configure_item(..., min_value=..., max_value=...)` 反映コードを追加。

**処理フロー（初回登録）**
1. `G.text(...)` 呼び出し → ParameterRuntime.before_shape_call()
2. FunctionIntrospector が `text` のメタ/シグネチャ取得
3. ValueResolver.resolve(): `merged` 構築 → `font_index` を `_resolve_scalar()` で処理
4. `_resolve_scalar()` が `TextRenderer.get_font_face_count(merged["font"])` を参照し、`RangeHint(0..N-1)` を注入
5. ParameterStore.register() により Descriptor 登録 → GUI 側で初期レンジが 0..N-1 で描画

**例外とフォールバック**
- `fontTools` 不在/取得失敗: N=1 とし、`font_index` は常に 0 のみ（`.ttf/.otf` と同等）。
- ファイル特定不可（検索失敗）: 既定フォント候補に対して同様のロジックを適用。最終的に失敗時は N=1。
- 実ファイル確定前に GUI を表示する場合: 本設計では初回 `text(...)` 検出時にレンジが確定（GUI マウント前に initialize() が一度 `user_draw(0.0)` を呼ぶため、多くのケースで間に合う）。

**パフォーマンス**
- フェイス数の取得は初回のみ実施し、`_face_count_cache` で再利用。
- `.ttc` の `TTCollection` 構築コストは軽微。失敗時は即時フォールバック。

**テスト方針（単体/結合）**
- 単体（モック）
  - `TextRenderer.get_font_face_count()` に対して、TTCollection をモックして `N` を返すケースと ImportError/例外時のフォールバック（N=1）を検証。
  - `ParameterValueResolver.resolve()` で `text.font_index` の Descriptor.range_hint.max が `N-1` になることを検証（`merged` に `font` を与える）。
- 結合（軽め）
  - 実フォント依存を避け、`TextRenderer.get_font_face_count` を monkeypatch して N=6 を返す → GUI の `font_index` スライダーが 0..5 で生成されることを確認（DPG の `get_item_configuration` を利用）。

**互換性**
- 公開 API 変更なし。UX のみ改善。
- 既存の `__param_meta__` はフォールバックとして残るため、他所への影響は最小。

**作業チェックリスト**
- [ ] `TextRenderer.get_font_face_count(font_name)` 実装（`.ttc`→N、その他→1、キャッシュ付き）
- [ ] `value_resolver.resolve/_resolve_scalar` を拡張し、`text.font_index` に動的 RangeHint を適用
- [ ] 既存メタは据え置き（将来削除検討）。必要なら `max: 32` のコメント補足
- [ ] 単体テスト（モック）追加: N=6 → max=5、ImportError/例外 → max=0
- [ ] 結合テスト（軽量）: monkeypatch で GUI スライダーの min/max を検証
- [ ] ドキュメント更新（本ファイルでの設計メモ、`docs/shapes.md` に注意書き追記）

**確認事項（要回答）**
- 既定フォントの解決順序は `TextRenderer.get_font()` と完全一致で問題ないか（部分一致の優先度など）。
- フォント変更に対するランタイム再構成（`update_descriptor_range` 追加）は今回スコープ外でよいか。
- ログ方針: フェイス数取得失敗時のログは `debug` に留め、ユーザー通知は行わない方針でよいか。

**将来拡張（任意）**
- `font_face_name`（文字列）指定を追加し、`font_index` を非推奨化（`.ttc` 内フェイスを名前で選択）。
- Parameter GUI にフォント/フェイス選択 UI（コンボ/ファイルブラウザ）を追加し、`font` を GUI 操作可能に（現状は `choices=[]` で非表示）。

