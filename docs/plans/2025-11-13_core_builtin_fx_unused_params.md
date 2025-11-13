どこで: engine/core/lazy_geometry.py の内蔵 fx（_fx_scale, _fx_rotate）
何を: 未使用引数（auto_center）を整理（削除 or 実装）。必要に応じて Geometry 側に中心計算補助を追加
なぜ: 署名と実装の乖離を無くし、読みやすさと保守性を向上するため

# 背景 / 現状

- 内蔵 fx の署名:
  - src/engine/core/lazy_geometry.py:320 `def _fx_scale(..., auto_center: bool = False, pivot: tuple[...], scale: ...)`
  - src/engine/core/lazy_geometry.py:333 `def _fx_rotate(..., auto_center: bool = False, pivot: tuple[...], angles_rad: ...)`
- 実装では `auto_center` は未使用（pivot のみ利用）。
- LazyGeometry のチェーン用糖衣:
  - src/engine/core/lazy_geometry.py:173 `def scale(..., center=(0,0,0))` → plan に `{"auto_center": False, "pivot": center, "scale": ...}` を積む
  - src/engine/core/lazy_geometry.py:191 `def rotate(..., center=(0,0,0))` → plan に `{"auto_center": False, "pivot": center, "angles_rad": ...}` を積む
- つまり auto_center は常に False で渡され、内蔵 fx 側でも未使用。署名だけが残存している状態。
- 参考: effects/affine は auto_center を意味ある形で実装済み（平均座標を中心に適用）。

# 目標（Goal）

- 署名と実装の乖離を解消。
- 方針を「削除」または「実装」のいずれかに統一（影響最小を優先）。

# 非目標（Non-Goals）

- API 層（api.effects / api.shapes）の振る舞い変更。
- effects/affine の挙動変更（auto_center の定義は維持）。

# 選択肢（提案）

案A（推奨・小さく安全）: 未使用引数を削除
- 変更:
  - _fx_scale/_fx_rotate から `auto_center` 引数を除去。
  - LazyGeometry.scale/rotate が plan へ積む dict から `"auto_center": False` を削除。
- 利点: 実装と署名が一致し、意味のない引数が消える。互換性影響は内部に限定。
- 注意: 既存 plan の直列生成コードは同ファイル内でのみ参照、外部影響ほぼなし。

案B（機能拡張）: auto_center=True を実装
- 変更:
  - _fx_scale/_fx_rotate 内で `auto_center=True` のときに形状中心を算出して pivot を上書き。
  - 中心計算の補助を Geometry に追加（名称例: `centroid()` または `center_of_mass()`）。
  - LazyGeometry.scale/rotate に `auto_center: bool=False` を公開引数として追加（任意）。
- 中心計算の定義（候補）:
  - シンプル: 全頂点の平均（effects/affine と同じ定義）。
  - 代替: バウンディングボックス中心（外れ値に強い）。今回は affine と整合させ平均を採用。
- 利点: 使い勝手向上（Lazy 側でも中心自動化が可能）。
- 注意: 公開シグネチャ追加になるため最小破壊を意識（デフォルト False で互換維持）。

# 推奨

- まずは 案A（削除）を採用し、署名と実装の整合性を回復（変更小）。
- ユースケースが固まれば、別 Issue/PR として案Bを検討（Geometry 補助メソッドの設計含む）。

# 作業手順（チェックリスト）

- [ ] 現状の使用箇所確認（`rg "auto_center" src/engine/core/lazy_geometry.py`）
- [ ] 案Aの適用: _fx_scale/_fx_rotate から auto_center を削除
- [ ] 案Aの適用: LazyGeometry.scale/rotate が積む dict から `"auto_center": False` を削除
- [ ] コメント整備: 便宜操作（scale/rotate）の説明を更新（pivot のみ）
- [ ] Lint/Format/Type（変更ファイル限定）: `ruff check --fix {path}` / `black {path}` / `isort {path}` / `mypy {path}`
- [ ] 影響低テスト: LazyGeometry 経由の scale/rotate が期待通りに動作するスモーク

（案B を採用する場合の追加手順）
- [ ] Geometry へ `centroid()`（=coords の平均）を追加（src/engine/core/geometry.py）
- [ ] _fx_scale/_fx_rotate で `auto_center=True` の場合 pivot を `g.centroid()` に置換
- [ ] LazyGeometry.scale/rotate に `auto_center: bool=False` を公開引数として追加（任意）
- [ ] docstring/architecture.md の注記追記（Lazy 便宜操作に auto_center オプションがある旨）

# 受け入れ基準（Acceptance Criteria）

- [ ] src/engine/core/lazy_geometry.py の _fx_scale/_fx_rotate から未使用引数が無くなる（案A）
- [ ] LazyGeometry.scale/rotate が積む plan から auto_center が消える（案A）
- [ ] 既存パイプライン構築・評価に挙動差がない（回帰なし）
- [ ] mypy/ruff/black/isort が対象ファイルでグリーン

# 影響範囲・リスク

- 低（案A）: 内部関数の署名整理と plan の不要項目削除のみ。外部 API 影響なし。
- 中（案B）: Geometry への補助追加と LazyGeometry 署名の公開変更。慎重な合意が必要。

# 事前確認事項（Open Questions）

- [ ] 今回は案Aで進めて良いですか？（小変更）
- [ ] 案Bの中心定義は「頂点平均」でよいですか？将来的に「線長重み付き平均」や「BBox中心」を選べる設計は必要ですか？
- [ ] LazyGeometry に auto_center を公開したい要望はありますか？現段階では effects/affine で十分でしょうか？

