# plan: `effects/fill.py` リファクタリング（挙動維持・行数削減）

目的
- 直近の修正で安定化した挙動（穴の維持・共通間隔・共平面化）を完全に維持しつつ、コードの重複と説明的コメントを整理し、可読性と行数を削減する。
- 公開 API／引数・`__param_meta__`・テスト挙動は不変。

非目標（今回やらない）
- 機能追加やチューニングの挙動変更（密度定義・偶奇規則・角度適用）。
- 依存追加や大規模モジュール分割。

作業方針
- 既存のロジックを「小さな関数」に抜き出して再利用し、同等の説明コメントを docstring に収容（冗長コメントは削減）。
- 可能な限り `fill.py` 内で完結（新規ファイルは最小限）。

チェックリスト
- [x] 平面フレーム決定を関数化
  - ` _choose_coplanar_frame(coords, offsets) -> (R, z, chosen_idx)`
  - リング優先→PCA フォールバックの現在実装を移設（重複/try 節を削減）。
- [x] 共通間隔計算の関数化
  - ` _global_spacing(ref_height_global, density) -> float`
  - ライン本数のクランプ（2..MAX）を一か所に集約。
- [ ] スキャンライン生成の共通化
  - ` _scanlines(min_y, max_y, spacing) -> np.ndarray`
  - `_generate_line_fill` と `_generate_line_fill_evenodd_multi` の間で共有。
- [ ] 角度回転の共通処理を小関数化
  - forward/ inverse 回転行列と中心の計算を `_rotation2d(center, angle)` に集約。
- [x] グルーピング補助の最小化
  - 代表点（rep）ベースの内包判定＋ログのみ残し、重心計算は必要箇所のみ呼ぶ。
  - on-edge 安定化の微小 eps は現状維持。
- [x] ログユーティリティの簡素化
  - デバッグログは削除（要望により）。
- [x] しきい値算出の共通化
  - ` _planarity_threshold(diag) -> float` を導入し、グローバル/ポリゴンで共有。
- [x] `_generate_line_fill` の間隔計算を `_global_spacing` に統一
  - 単一ポリゴン経路でも見かけ密度が一致。
- [ ] docstring とコメントの整理
  - 目的/前提/返り値のみを簡潔に。重複説明は削除。
- [x] 変更範囲の高速チェック
  - ruff/black/isort/mypy: 変更ファイルのみ。
  - pytest: 関連テスト（fill 系・文字/順序/回転ケース）優先。

影響/リスク
- 関数抽出により分岐条件の取り違えが起こり得るため、段階的に小さく適用しテストで検証する。
- Numba デコレータ部は配置を変えない（関数抽出は Python 側ヘルパ中心）。

検証対象（優先）
- 0/9/9o/o9/! i j（小島）
- affine の X/Y/Z 回転を含むケース
- `density/angle/angle_sets` の配列サイクル（tests に既存）

オプション（要相談）
- [x] `_build_evenodd_groups` を `util/` へ移動（他エフェクトでも再利用可）。
- [x] 共平面フレーム選択を util 化し、fill からも利用（`util/geom3d_frame.py`）。
- [ ] PCA フォールバックを軽量 SVD に統一（現在の NumPy 実装で十分なら据え置き）。

完了条件（DoD）
- 既存テストに加えて 9o/o9・単独 f/s の目視確認を通過。
- 行数は `fill.py` で 10–20% 減（目安）かつ可読性向上。

質問（確認したい点）
- オプションの util への小分割は許容しますか？（行数は `fill.py` 減、総行数は微増の可能性）
- ログ量は現在の粒度を維持で良いですか？（`PXD_FILL_DEBUG=1` 時のみ）
