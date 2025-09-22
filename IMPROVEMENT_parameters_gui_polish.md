# Parameters GUI 改善チェックリスト（enum/レイアウト/操作性）

目的

- スクリーンショット[prams_gui.png]で観測された可読性・操作性の課題を解消し、短時間で使いやすい UI に仕上げる。
- 実装は `engine/ui/parameters/` 層に限定（Effects/Shapes 本体は非対象）。

スコープ

- enum 表示（セグメント/ドロップダウン）、スライダー表示、ラベル/グルーピング、操作性（KB/マウス）、視認性（配色）
- 既存の `ParameterStore/Descriptor/RangeHint` と `FunctionIntrospector` の利用前提（API 変更なし）

非スコープ

- 新しい外部依存の導入、描画基盤の総入替、MIDI 連携強化、国際化（ラベル翻訳）

---

タスク一覧（着手順の推奨）

A. Enum 表示の改善（最優先）

- [x] 幅配分をテキスト幅ベースにする（長いラベルが潰れない）
  - 測定: `pyglet.text.Label` の content width を使い、左右パディングを加味して各セグメント幅を決定
  - 参考: `src/engine/ui/parameters/panel.py` の `EnumWidget._update_layout()`
- [ ] 合計必要幅 > 可用幅のときドロップダウンへ自動フォールバック
  - 新規 `DropdownWidget`（単純ポップアップ）を追加し、`_create_widget()` で選択
  - フォールバック判定はピクセルベース（例: 必要幅がエリアの 90% 超過）
- [ ] セグメントの視認性向上（コントラスト/枠線/パディング）
  - 選択: 枠線+明色、非選択: 彩度/明度をさらに落とす、角丸
  - 最小ヒット領域（幅/高さ）とセグメント間マージンを確保

B. スライダー/値ラベルの整理

- [ ] スライダーの薄いトラックを再導入（位置の相対感を可視化）
  - 参考: `SliderWidget._ensure_graphics()` の `self._track = None` を見直し
- [ ] 値ラベルの配置最適化（enum には表示しない/重ね表示の採用）
  - 数値系: 右側 or 上側に小さめ表示、enum: セグメント内のテキストのみ

C. ラベリング/グルーピング

- [ ] 関数/インデックス単位の見出し行を挿入（例: `affine#0`）
  - 子要素は短い param 名（`angles.x` 等）のみ表示
  - 参考: `ParameterPanel.update_descriptors()` と `layout()` で見出し行を組み込む
- [ ] 並び順を scope→name→index→param に安定化（または生成順を採用）
  - 参考: `src/engine/ui/parameters/window.py` の `refresh()` 内ソート

D. 操作性（KB/マウス）

- [ ] Enum のキーボード操作: ←/→ で移動、数字キー 1..9 で直接選択
  - `ParameterWindow` の key イベントで `EnumWidget` へ転送
- [ ] フォーカスリング表示（現在操作対象のウィジェットを枠線で示す）
- [ ] Reset の発見性向上（右クリックメニュー or リセットアイコン）
  - 既存: Cmd/Accel+クリックで reset（見た目に表示が無い）

E. 情報表示（単位/ヒント/変更状態）

- [ ] 単位/補足のツールチップ（`help_text` を hover で表示）
  - 参考: `ParameterDescriptor.help_text` は既に格納済み
- [ ] 整数/step の視覚化（値の丸めルールを補助表示）
- [ ] 既定値からの変更マーカー（ラベルにドット or 色）
  - 現値と `default_value` 比較で判定（`ParameterStore.current_value()`）

F. フォールバック/異常値ハンドリング

- [ ] enum 値が `choices` 外だった場合は「未サポート」状態を明示（グレー表示+バッジ）
  - 選択操作により最寄りの選択肢へ遷移可能、ただし自動変換は行わない

G. 設定/レイアウト調整

- [ ] `ParameterLayoutConfig` にコンパクト/標準のプリセット（row_height/padding/font_size）
- [ ] 列幅比（ラベル:ウィジェット:値）の定数化と型ごとのチューニング

H. パフォーマンス/安定化

- [ ] EnumWidget の再構築を「choices 変更時」に限定（毎フレーム再生成を回避）
- [ ] Panel のレイアウト計算を viewport 変更時に限定（現在は十分軽量だが念のため）

I. テスト/検証

- [ ] ヘッドレス（CI）での描画なし検証（レイアウト計算のみ）
  - `EnumWidget` の幅割り当て/フォールバック判定の単体テスト
- [ ] `ValueResolver` との連携: enum の `supported=True` と `choices` 伝播を確認
- [ ] 最小スモーク: `offset.join`/`twist.axis`/`extrude.center_mode`/`text.align`/`fill.mode` が GUI で操作可能

---

受け入れ条件（DoD）

- 長い enum ラベル（例: polyhedron の `polygon_type`）が潰れず読める（セグメント幅調整 or ドロップダウンへ自動移行）
- ラベルが簡潔（見出し+短い param 名）で、並びが安定
- スライダーにトラックが表示され、値位置が直感的
- Enum の選択状態が明瞭（コントラスト向上）、ヒット領域が十分
- 既定値からの変更が視覚的に分かる
- 変更ファイル限定の `ruff/black/isort` 緑、ヘッドレス単体テストが通る

備考

- 既存の `ParameterDescriptor.help_text/choices` と `RangeHint` を最大限活用し、API 変更なしで実装可
- ドロップダウンは最小限機能（クリックで開閉/スクロール/クリック選択）から開始し、必要に応じてキーボードを追加

関連ファイル（参照）

- `src/engine/ui/parameters/panel.py`
- `src/engine/ui/parameters/window.py`
- `src/engine/ui/parameters/state.py`
- `src/engine/ui/parameters/value_resolver.py`
