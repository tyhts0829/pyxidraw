# ADR 0001: 多面体データ形式を pickle から npz へ移行

- ステータス: Accepted
- 日付: 2025-09-04

## 背景 / コンテキスト

`shapes/polyhedron.py` は正多面体の頂点データを `data/regular_polyhedron/` 配下から読み込みます。従来は
`*_vertices_list.pkl`（pickle）を使用していましたが、以下の課題がありました。

- 安全性: pickle はロード時に任意コード実行の危険がある。
- 互換性: Python 実装/バージョン/クラス定義に依存が強く、長期保守に不利。
- 再現性: データスキーマ（dtype/shape/順序）が不透明になりがち。

## 決定

データ形式を `.npz`（NumPy zipped archive）へ移行し、これを正とする。

- スキーマ: `arr_0, arr_1, ...` の連番キー。各配列は `float32`、形状は `(N_i, 3)`（XYZ）。
- 読み込み: `np.load(path, allow_pickle=False)` を前提に、`arr_*` を昇順で読み出す。
- 互換: 当面は `arrays` キー（配列リスト）も受理。旧 `.pkl` は暫定互換で読み込みつつ、その場で `.npz` を自動生成。

## 影響 / トレードオフ

メリット:
- 安全性の向上（任意コード実行リスクの排除）。
- 環境非依存で将来の移行コストが低い。
- スキーマが明示で検証容易（`float32`/`(N,3)` 決定的）。
- 実装の単純化（ローダ/互換レイヤの縮小）。
- zip 圧縮により配布効率が良い（CI アーティファクト/キャッシュ向き）。

デメリット/注意点:
- `.npz` は展開コストがある（メモリマップは `.npy` 向け）。巨大データでは `.npy` 分割も検討余地。

## 代替案と却下理由

- そのまま pickle 継続: 安全性と互換性の問題が残るため却下。
- `.npy` 複数ファイル: メモリマップは有利だが、管理が煩雑（ファイル数が増える）。
- JSON/CSV: 浮動小数の精度/サイズ/読み込み速度で不利。
- HDF5: 機能は豊富だが依存が増え、本件の規模では過剰。

## マイグレーション計画

1. 変換スクリプトを配布: `scripts/convert_polyhedron_pickle_to_npz.py`
2. 実行例:
   - ドライラン: `python scripts/convert_polyhedron_pickle_to_npz.py --dry-run --verbose`
   - 変換: `python scripts/convert_polyhedron_pickle_to_npz.py --verbose`
   - 削除: `python scripts/convert_polyhedron_pickle_to_npz.py --delete-original`
3. 段階廃止: `.pkl` ローダは告知期間後に削除予定。

## 参照

- `PROPOSAL_BREAKING_CHANGES.md` 「決定記録: polyhedron データ形式（pickle → npz）」
- `shapes/polyhedron.py`（npz 優先/自動移行の実装）
- `scripts/convert_polyhedron_pickle_to_npz.py`

