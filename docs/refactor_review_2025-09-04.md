# リファクタリング・レビュー（2025-09-04）

目的: コード全体を俯瞰し、「美しく・シンプルで・可読性が高い」観点で気になる箇所を洗い出し、具体的な改善案（優先度つき）を提示する。

進捗チェック（2025-09-04 現在）
- [x] 1) util/geometry: 近似/厳密の混在対策（geometry_* を非推奨化・警告付与）
  - [x] DeprecationWarning と用途説明を追加
  - [x] 呼び出し箇所の完全置換（不要: 現状参照なし）
- [x] 2) engine/io/controller: 終了処理の一貫化（exit→例外）
  - [x] `InvalidPortError` 新設、handle_invalid_port_name で raise
  - [x] runner 側で捕捉して `SystemExit(2)`（CLI 相当）
- [x] 3) effects/weave/fill: 可読性向上
  - [x] 冒頭にアルゴリズム概要コメントを追加
  - [ ] 関数分割（交点計算/候補線生成/緩和ステップの段落化）
- [x] 4) common/cacheable_base: 仕様の明文化
  - [x] 前提（hashable）/追い出し/LRU/環境変数/デバッグ指針を docstring に追記
- [x] 5) effects/extrude: スケール中心の選択肢追加
  - [x] `center_mode=("origin"|"auto")` 実装（既定は互換維持）
  - [x] README/cheatsheet に追記
- [x] 6) engine/core/transform_utils: 役割の明確化
  - [x] 「Geometry メソッド推奨」「transform_combined 主用途」を明記
- [x] 7) data/* 生成スクリプトのログ
  - [x] `icecream` → `logging` へ置換（実行時のみ影響）

## 結論サマリ
- 現状は設計がシンプルに整理され、可読性も高い。特に `Geometry` 一本化・関数エフェクト・単層キャッシュは良好。
- 追加で磨けるポイントは「重複/近似機能の統合」「長関数の分割」「エラーハンドリングの一貫化」「軽いガイドの追補」。
- どれも小～中スケールの変更で実現でき、リスクは低い。

## 気になるポイント（具体例）

### 1) util/geometry: 近似機能の二重実装（統合提案）
- [x] 簡易版（geometry_*）に DeprecationWarning と用途説明を付与
  - [x] 呼び出しの完全置換／厳密版へ統一（不要: 現状参照なし）
- 事象:
  - `transform_to_xy_plane()/transform_back()`（Numba最適化）と、`geometry_transform_to_xy_plane()/geometry_transform_back()`（簡易版）が並存。
  - 後者は重心を z=0 に落とすだけで、意味的に別物。呼び分け基準が読み手に伝わりにくい。
- 改善案（優先度: 中）:
  - `geometry_*` 系は非推奨にして docstring に用途を明記、もしくは内部で `transform_*` にフォールバックして意味を揃える。
  - 片方に統合し、呼び出し箇所を `transform_*` に寄せる（CI で使用箇所を検出）。

### 2) engine/io/controller: 終了処理の一貫化
- [x] `exit(1)` を例外送出に変更（`InvalidPortError`）
- [x] runner 側で捕捉し `SystemExit(2)` へマッピング
- 事象:
  - `handle_invalid_port_name()` で `exit(1)` を直接呼ぶ。ライブラリ層としては例外送出の方が一貫。
- 改善案（優先度: 中）:
  - 専用例外 `InvalidPortError` を挙げ、呼び出し側（CLI/ランナー）で終了コードを決定。
  - 例外メッセージに候補ポート一覧を含める今の振る舞いはそのまま活かす。

### 3) effects/weave, effects/fill: 長関数の分割と命名整理
 - [x] 冒頭にアルゴリズム概要コメントを追加
 - [x] weave: 高レベル手順を `_webify_single_polygon()` に抽出（内部は既存 njit 利用）
 - [x] fill: 高レベル手順を `_fill_single_polygon()` に抽出
 - [ ] さらなる段落化（numba 関数の粒度は現状維持）
- 事象:
  - `webify.py` は多段の njit 関数 + 高レベル関数が 1 ファイルに密集。`filling.py` も複数の生成系が同居。
- 改善案（優先度: 中）:
  - 役割ごとに小さな関数へ分割（例: 交点計算、候補線生成、緩和ステップなど）。
  - ファイル先頭に「アルゴリズム概要」の 5 行程度の解説コメントを追加（保守者の学習コスト削減）。

### 4) common/cacheable_base: キャッシュ仕様の明文化
- [x] 前提・追い出し戦略・環境変数制御・デバッグ指針を docstring に明記
- 事象:
  - `LRUCacheable` は便利だが「引数が hashable であること」「 NumPy 配列は想定外」など暗黙の前提がある。
- 改善案（優先度: 低）:
  - docstring に前提と落とし穴（例: 可変オブジェクトのキー、メモリ使用の上限/追い出し戦略）を明記。
  - 可能なら `@lru_cache` レイヤのキー生成を明示化（`functools.cache` 相当の軽ラッパ）してデバッグ性を向上。

### 5) effects/extrude: スケールの中心
- [x] `center_mode=("origin"|"auto")` を追加（互換既定）
- [x] README/cheatsheet に追記
- 事象:
  - 押し出し側のスケーリングが原点基準（`(line + extrude_vec) * scale`）になっており、直感的には「押し出し後の線の重心基準」でのスケールが自然な場面がある。
- 改善案（優先度: 低）:
  - `center: Vec3 | Literal["auto"]` を追加し、`"auto"` の場合は押し出し先ラインの重心を中心にスケール。
  - 既定は互換性維持（現状のまま）。

### 6) engine/core/transform_utils: Geometry メソッドとの重複
- [x] 「Geometry メソッド推奨」「transform_combined 主用途」を明記
- 事象:
  - `translate/scale/rotate_*` が `Geometry` メソッドと重複。学習コストは小さいが、読み手が迷い得る。
- 改善案（優先度: 低）:
  - `transform_utils` は「まとめて適用」の `transform_combined()` を主用途に、個別関数は一文で `Geometry` メソッド推奨のコメントを追加。

### 7) data/* 生成スクリプトのログ
- [x] `icecream` → `logging` に置換（実行時のみ影響）
- 事象:
  - `icecream` が `if __name__ == "__main__"` 節で使われている。実行時依存ではないが、本体ポリシー（engine 層は logging）に合わせると美しい。
- 改善案（優先度: 低）:
  - 可能なら `logging` に置き換え、`__main__` のみで設定。

## 「美しさ・シンプルさ・可読性」観点での原則（このリポジトリ向け）
- 単一データモデル（Geometry）を中心に、「関数（純粋）→パイプライン（組立）→描画/入出力（副作用）」の三層を徹底。
- 0..1 正規化は `common/param_utils` に集約し、表記のブレを避ける。
- 型エイリアス（Vec2/Vec3）で引数の意味を表す（ドキュメントより短く有益）。
- 長い関数は「データ準備」「コア計算」「結果整形」に段落化して 30～40 行程度を目安に分割。
- 例外処理はライブラリ層で raise、アプリ/CLI 層で exit（責務分離）。

## 推奨アクション（小さく速く）
1. util/geometry の `geometry_*` 系を非推奨化 or `transform_*` に置換（済: 非推奨化・警告）。
2. engine/io/controller の `exit(1)` → 例外送出（済）＋ runner 側終了コード処理（済）。
3. effects/weave/fill の冒頭にアルゴリズム概要コメント（済）＋必要に応じて関数分割（未）。
4. cacheable_base に前提と落とし穴を docstring（済）。
5. extrude に `center="auto"` オプション（済）。

---
最小の改善でも「読みやすさ」は確実に向上します。必要なら、上記 1～3 をまとめてパッチ化します。
