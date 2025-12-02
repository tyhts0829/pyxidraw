# Parameter GUI テキスト高さ反映計画

- [ ] 仕様確認: `param_meta` の `height` をテキスト入力の高さとして優先し、未指定時のみレイアウト (`ParameterLayoutConfig.row_height`) にフォールバックするでよいか？単位は現行と同じ px で問題ないか。
- [ ] 現状整理: `text` の `__param_meta__` で `height` を設定 → `value_resolver` が `string_height` へ引き渡し済み → `dpg_window_content._create_widget` で `string_height` を無視し `row_height` を高さに利用している点を再確認する。
- [ ] 実装方針: `_create_widget` で文字列入力生成時に `desc.string_height` を優先適用し、値が 0/None のときだけレイアウト既定を使うようにする（multiline 以外への影響は避ける）。必要なら最小値のクランプを入れる。
- [ ] 影響確認: 他の string パラメータ（palette など）で高さ指定がない場合の見た目が変わらないことを確認する。
- [ ] 動作確認: `text` パラメータで `height` を変えたときに GUI の入力ボックス高さが変わることを目視で確認する。必要なら最小限の `ruff/black` を変更ファイルに実行。

質問/確認したいこと:
- 上記のように「meta の height 優先、未指定は既定高さ」運用でよいか？
- 最小/最大高さのガードは設けるべきか（例: 16px 以上など）？不要ならそのまま適用します。
