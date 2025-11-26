# palette ディレクトリの使い方（コピペ利用向け）

この `palette` ディレクトリを別のジェネレーティブコーディングプロジェクトへそのままコピーして使う場合のガイドです。

## 前提
- Python 3.10+
- 追加依存なし（標準ライブラリのみ）で動作します。

## 1. ディレクトリを配置
プロジェクト内に `palette/` を丸ごとコピーしてください。パスが通るように、スクリプトと同じ階層か `PYTHONPATH` 上に置きます。

## 2. 最小サンプル（CLI/スクリプト内）
```python
from palette import ColorInput, generate_palette, PALETTE_TYPE_OPTIONS, PALETTE_STYLE_OPTIONS, export_palette

# ベース色
base = ColorInput.from_hex("#6699cc")

# ラベル→Enum の辞書は option list を dict 化して取得
palette_type = dict(PALETTE_TYPE_OPTIONS)["Complementary"]
palette_style = dict(PALETTE_STYLE_OPTIONS)["Square"]

palette = generate_palette(
    base_color=base,
    palette_type=palette_type,
    palette_style=palette_style,
    n_colors=4,
)

# sRGB (0-1) の色リストを取得
colors = export_palette(palette, "srgb_01")
print(colors)
```

## 3. 出力フォーマット
`export_palette(palette, fmt)` の `fmt` は以下を指定できます。
- `"srgb_01"`: `(r, g, b)` in [0, 1]
- `"srgb_255"`: `(r, g, b)` in 0–255 ints
- `"hex"`: `"#rrggbb"`
- `"oklch"`: `(L, C, h)`

Enum で扱いたい場合は `from palette import ExportFormat` で読み込み、`ExportFormat.SRGB_01` などを渡してください。

## 4. UI との統合（Dear PyGui など）
UI のコンボボックス用に、ラベル/Enum のペアを用意しています。
- `PALETTE_TYPE_OPTIONS`: `[(label, PaletteType), ...]`
- `PALETTE_STYLE_OPTIONS`: `[(label, PaletteStyle), ...]`
- `EXPORT_FORMAT_OPTIONS`: `[(label, ExportFormat), ...]`

`dict(PALETTE_TYPE_OPTIONS)[label]` のようにして Enum を取得できます。`example_dpg.py` に Dear PyGui との統合サンプルがあります（Dear PyGui が未インストールの場合はメッセージのみ表示）。

## 5. テスト
コピー後に動作確認する場合は、プロジェクトルートで:
```bash
python -m pytest
```
（pytest が無い場合は `pip install pytest` で導入）
