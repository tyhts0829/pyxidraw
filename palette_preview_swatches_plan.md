# palette プレビュー HEX テキスト → 色スウォッチ表示計画（ドラフト）

目的: Palette セクションの「Palette HEX: #xxxxxx, ...」という文字列表示をやめて、実際の色を横一列のスウォッチとして表示する。各スウォッチをクリックすると、その色情報（HEX）をクリップボードにコピーできるようにする。

## 現状把握

- プレビュー構築箇所: `src/engine/ui/parameters/dpg_window_content.py`
  - `build_palette_controls` の末尾で
    - `self._palette_preview_id = dpg.add_text("Palette: (未計算)")`
    - `Copy HEX` ボタンを配置。
  - `_refresh_palette_preview` で
    - `engine.ui.palette.helpers.build_palette_from_values(...)` で `palette.Palette` を生成
    - `palette.ui_helpers.export_palette(..., ExportFormat.HEX)` で HEX リストを取得
    - ` "Palette HEX: ..."`` というテキストを組み立てて  `dpg.set_value(self.\_palette_preview_id, text)`。
- HEX コピー機能:
  - `_on_palette_copy_hex` で `export_palette(..., HEX)` を使って改行区切り文字列をクリップボードへコピー。

## 方針（UI 表現）

- テキスト行を「スウォッチ群 + 簡易テキスト」に置き換える。
  - Dear PyGui の `color_button` または `drawlist`/`rect` ではなく、まずは `color_button` を使うシンプルな実装を目標。
  - 各スウォッチはクリック非対応（decorative）か、クリックで HEX コピーなどの拡張は後回し。
- レイアウト:
  - `Palette` ヘッダ内部に水平 `group` を作り、その中にスウォッチ群を追加。
  - 色数が多い場合は折り返し無しで横スクロールではなく、そのまま横並び（実際のパレット数は通常 4〜8 を想定）。

## やること（チェックリスト）

### 1. スウォッチ用コンテナ/タグ設計

- [ ] `ParameterWindowContentBuilder` にスウォッチ用コンテナタグを保持するフィールドを追加
  - 例: `self._palette_swatches_container: int | str | None = None`
- [ ] `build_palette_controls` 内でプレビュー用グループを拡張
  - 既存の `group(horizontal=True, parent=pal_hdr)` を
    - 左: プレビュー用スワッチコンテナ
    - 右: `Copy HEX` ボタン
      に分離するイメージに変更。
  - `self._palette_swatches_container = dpg.add_group(parent=pal_hdr, horizontal=True)` のような形を検討。

### 2. `_refresh_palette_preview` の役割変更

- [x] 現在のテキスト更新ロジックを「スウォッチ再構築」に置き換える/併用する  
  - `export_palette(palette_obj, HEX)` で HEX リストを取得。
  - ループで HEX → RGBA(0..1) → 0–255 に変換し、スウォッチウィジェットを追加。
- [x] 既存の `self._palette_preview_id` テキストは実質的に使わず、スウォッチのみを表示する。
- [x] 再描画時のハンドリング
  - `_refresh_palette_preview` が呼ばれるたびに
    - 既存スウォッチを削除する（`dpg.delete_item` または `dpg.delete_item(children_only=True)` 相当）か、
    - 既存個数との差分だけ追加/更新する。
  - スロットルは不要（Palette 変更頻度は低い想定）。

### 3. スウォッチウィジェットの選定と実装詳細

- [x] Dear PyGui のどのウィジェットを使うかを確定  
  - 優先: `dpg.add_color_button`（RGBA を渡すだけでボタン表示）。
  - ボーダーやサイズは最小限にし、`self._layout.row_height` などから適当に計算（横幅は行高さの 2 倍程度）。
- [x] 具体的な追加イメージ:
  - `dpg.add_color_button(parent=self._palette_swatches_container, default_value=[r,g,b,a], no_border=True, width=row_height*2, height=row_height, callback=...)`
  - スウォッチ間の水平スペースは既存の `item_spacing_x` に準拠。

### 4. HEX コピーとの整合（スウォッチクリック）

- [x] 既存の `Copy HEX` ボタンと `_on_palette_copy_hex` を削除する。  
- [x] 各スウォッチ（`color_button`）にクリックコールバックを設定し、その色の HEX 文字列をクリップボードへコピーするようにする。  
  - HEX への変換は `palette.ui_helpers.export_palette(..., HEX)` の結果、または `util.color` の補助関数を利用。  
  - クリップボードへの書き込みは `dpg.set_clipboard_text` を使用し、存在しない場合は何もしないフェイルソフトにする。

### 5. Snapshot/`api.C` との関係

- [x] スウォッチは純粋に UI 表示のみであり、`util.palette_state` や `api.C` には触れないようにする。
  - すでに `_refresh_palette_preview` 内で Palette オブジェクトを更新しているため、それを流用するだけでよい。
  - スウォッチ側で余計な状態を持たない（Palette の単一真実源は `util.palette_state` + `palette_obj`）。

### 6. テスト/確認

- [ ] 手動確認:
  - L/C/h や type/style/n_colors を変えるとスウォッチの色が追随すること。
  - 色数が変わったとき（例: n_colors=3→7）にスウォッチ数も更新されること。
  - 任意のスウォッチをクリックすると、その HEX がクリップボードへコピーされること。
- [ ] 自動テストは最小限:
  - 直接 DPG を触るテストは追加せず、スウォッチ構築ロジックを個別関数に切り出す場合のみ、その関数の引数 → 出力（スウォッチ数や色のリスト）の単体テストを検討。

### 7. ドキュメント/メモ更新

- [ ] `palette_base_color_lch_plan.md` または `palette_integration_plan.md` に「HEX テキスト → スウォッチ表示」への変更を追記。
- [ ] 必要なら README の Palette セクションのスクリーンショット/説明を更新。

---

メモ:

- 初回はシンプルに `color_button` ベースで実装し、見た目の細かい調整（丸角/枠線/サイズなど）は後から行う。
- 実装に入る前に、スウォッチクリック時の挙動（何もしない/HEX コピー/他）について追加要望があれば反映する。
