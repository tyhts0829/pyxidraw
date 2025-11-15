# dash エフェクト: 可変パターン対応 改善計画

## ゴール
- `dash_length` / `gap_length` に `list[float]` を受け入れ、例 `[1, 3, 2]` のようなサイクルパターンで破線化できるようにする。
- 既存のスカラー指定の挙動とパフォーマンス特性を維持する。

## 想定仕様（案）
- [x] `dash_length`, `gap_length` は `float | list[float] | tuple[float, ...]` を受け付ける。
- [x] スカラーは長さ 1 のパターンとして扱い、現行のダッシュ長・ギャップ長と等価にする。
- [x] `list` / `tuple` 指定時は `(dash[i], gap[i])` を順番に適用し、末尾まで行ったら先頭に戻るサイクルとする。
- [x] `dash_length` と `gap_length` の長さが異なる場合は、短い方を循環させて補完する（`i % len(seq)`）。
- [x] 非有限値または `dash[i] + gap[i] <= 0` を含む場合は、現行仕様にならい no-op（入力コピーを返す）とする。

## 実装タスク案
- [x] `src/effects/dash.py` の `dash` シグネチャを `float | Sequence[float]` 相当に拡張し、入力正規化ヘルパー関数を追加する。
- [x] 正規化で `dash_length` / `gap_length` を `np.ndarray`（float64）に変換し、`_count_line` / `_fill_line` に `dash_lengths`, `gap_lengths` として渡すようにする。
- [x] `_count_line` を「弧長配列 `s` を計算 → サイクルパターンで `pos` を進めながら各ダッシュ区間の頂点数を積算する」形に書き換える。
- [x] `_fill_line` を同様のサイクルパターンで `start` / `end` を計算し、既存の補間ロジックを繰り返し適用する形に書き換える。
- [x] 「線が短い」「パターンが 1 要素のみ」「全長が 0」「非有限値」などのエッジケースを再確認し、現行の no-op 条件を保つ。

## テスト・ドキュメント
- [x] `tests/effects/test_dash_basic.py` に list パターン用のケースを追加（例: `[1.0, 3.0, 2.0]` サイクルでの開始/終端座標検証）。
- [x] float と list 混在（例: `dash_length=[1, 2], gap_length=3.0`）の挙動をテストで固定する。
- [x] `docs/effects_arguments.md` と `src/effects/dash.py` の docstring を新仕様に合わせて更新する。
- [x] 公開 API スタブ `src/api/__init__.pyi` を再生成し、`dash` の引数型コメントを更新する。

## 要確認事項（ユーザーと相談したい点）
- [ ] `dash_length` / `gap_length` のどちらか一方だけを list 指定した場合の仕様（もう一方をスカラーとして循環補完でよいか）。
- [ ] GUI からは従来どおりスカラーのみを編集対象とし、list 指定はスクリプト/API 上級者向け機能とする方針でよいか。
- [ ] サイクル中のどこから開始するかを固定（常にインデックス 0 開始）で問題ないか。
