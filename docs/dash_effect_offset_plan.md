# dash エフェクト: offset パラメータ追加 改善計画

## ゴール
- `dash` エフェクトに `offset` 引数を追加し、破線パターンの開始位相を制御できるようにする。
- `offset` はスカラーおよび `list` / `tuple` を受け付け、既存の挙動（常に実線から開始）との後方互換性を維持する。

## 想定仕様（案）
- [x] `offset` 引数を追加し、型は `float | list[float] | tuple[float, ...]` とする。
- [x] 既定値 `offset=0.0` のとき、現行実装と完全に同じ結果（先頭が常に実線）になる。
- [x] `offset` は「パターンの開始位置（[mm]）」として解釈し、`dash_length` / `gap_length` のサイクル上で `offset` 分だけ進めた位置から描画を開始する。
- [x] `dash_length` / `gap_length` と同様、`offset` が配列の場合は図形ごと（行ごと）にサイクルして適用する。
- [x] 非有限値または負の `offset` が与えられた場合は、0 へクランプする（例外は送出しない）。
- [x] `dash_length` / `gap_length` の組と `offset` を合わせた結果、全区間がギャップになるようなケースでも、元の線を no-op として返す。

## 実装タスク案
- [x] `src/effects/dash.py` の `dash` シグネチャに `offset: float | list[float] | tuple[float, ...] = 0.0` を追加し、`_as_float_seq` と同様の正規化を行う。
- [x] 正規化済み `offsets` を `np.ndarray` 化し、`dash_lengths` / `gap_lengths` と同様にサイクル適用できるようにする。
- [x] `_count_line` に `offset` を引数追加し、パターン座標（u軸）上で `offset` を考慮したカウントロジックを実装する（u軸のダッシュ区間を [offset, L+offset] と交差させる）。
- [x] `_fill_line` も同様に `offset` を受け取り、u軸のダッシュ区間を [offset, L+offset] と交差させた結果を線分に変換するよう拡張する。
- [x] 既存のパス（offset=0 のとき）のパフォーマンスと挙動が大きく変わらないよう、ループ構造をシンプルに保つ。
- [x] `dash.__param_meta__` に `offset` のメタ情報を追加し、RangeHint を 0〜100 に設定する。

## テスト・ドキュメント
- [x] `tests/effects/test_dash_basic.py` に `offset` 用のテストケースを追加する。
  - 例: `dash_length=3, gap_length=2, offset=0` → 既存テストと同じ結果。
  - 例: `dash_length=3, gap_length=2, offset=2` → 先頭 2mm がギャップ扱いとなり、最初の実線開始位置がずれることを検証。
  - 例: `dash_length=[1,3,2], gap_length=2.0, offset=1` のような「パターン＋offset」の組み合わせ挙動を固定。
- [ ] `offset` を list/tuple で与えた場合のサイクル挙動（例: `offset=[0, 1]`）をテストで固定する。
- [x] `docs/effects_arguments.md` に `offset` 引数を追記し、意味（パターン開始位置）と単位を説明する。
- [x] `src/effects/dash.py` の docstring に `offset` の説明を追記する。
- [x] 公開 API スタブ `src/api/__init__.pyi` を更新（`python -m tools.gen_g_stubs`）し、`dash` のシグネチャに `offset` を反映させる。

## 要確認事項（ユーザーと相談したい点）
- [ ] `offset` の適用単位を「ジオメトリ全体で共通」とするか、「各ポリラインごとに独立」とするか（現状の線ごとの処理フローに合わせる場合、行ごと独立の方が自然）。
- [ ] 負の `offset` やパターン長を大きく超える値の扱い（単純に `offset % T` へ正規化でよいか）。
- [ ] `offset` を GUI から操作対象にするか（RangeHint の上限、UI の複雑さ）それともスクリプト/API 専用パラメータとするか。
- [ ] `offset` が大きく、最初の数パターンが完全にスキップされる状況での期待挙動（先頭にギャップのみが続くのを許容するか、少なくとも 1 本はダッシュを保証するか）。
