% `clip` 結果キャッシュ + ダイジェスト量子化 + モード安定化 計画

目的
- ユーザー側のスケッチに手を入れずに、`clip` の毎フレーム計算を大幅に削減する。
- 既存の「マスク前処理キャッシュ」に加えて、「マスク×対象の結果（Geometry）」を LRU で再利用。
- キーの安定化（ダイジェスト量子化）とモード安定化により、キャッシュヒット率を現実的に高める。

非目標
- 出力の近似化や品質劣化（線の欠落・形状差異など）は避ける。
- ユーザー API の変更、draw コードの修正要求。

背景（現状）
- すでに「マスクの前処理（Shapely region / 投影XYリング・グリッド）」の LRU は導入済み。
- しかし最も重いのは「マスク×対象のクリップ本体」であり、ここは毎フレーム実行されるため CPU 負荷が大きい。
- digest は float32 生バイト基準で完全一致のため、微小ゆらぎでミスしやすい。

設計（提案）
1) 結果 LRU（内容ベース）
   - キー: `(mode, mask_digest_q, target_digest_q, draw_inside, draw_outside, draw_outline)`
     - mode: `"planar" | "proj"`（共平面/投影）
     - `mask_digest_q`, `target_digest_q`: 量子化後に blake2b-128 で生成。
   - 値: `Geometry`（出力）
   - 制限: `maxsize` と `max_verts` で保護（巨大出力は保存しない）。
   - LRU は `OrderedDict`。プロセス内限定、ワーカ単位で独立。

2) ダイジェスト量子化（安定化）
   - 量子化 step（例: `1e-4`）で coords を丸めてから blake2b。
   - 既定: `PXD_CLIP_DIGEST_STEP` 未指定なら `1e-4`。`common.settings` があればそちら優先。
   - 目的: 座標の微小差（回転/スケール・丸め）でのキャッシュミスを低減。

3) モード安定化（planar/projection の揺れ抑制）
   - マスクダイジェスト単位で「一度判定した mode を優先」する弱いピン止め（env で ON/OFF）。
   - 例: `PXD_CLIP_MODE_STABLE=1` 有効時、mask_digest ごとに最後の mode を記憶し、閾値付近での揺れを避ける。
   - 安全性: 既存の閾値判定が大きく変わる場合は上書き可能（最新判定を優先）。

4) 適用位置
   - `clip()` 内の planar/projection 分岐直後、実計算前に「結果 LRU」参照 → ヒット時は Geometry を返す。
   - ミス時は通常計算 → 出力が `max_verts` 以内なら LRU へ格納。

5) 環境変数（提案）
   - `PXD_CLIP_RESULT_CACHE_MAXSIZE`（既定 16）: 結果 LRU の上限件数。
   - `PXD_CLIP_RESULT_CACHE_MAX_VERTS`（既定 200_000）: 保存対象の最大頂点数閾値。
   - `PXD_CLIP_DIGEST_STEP`（既定 1e-4）: ダイジェスト量子化の step。
   - `PXD_CLIP_MODE_STABLE`（0/1, 既定 1）: モード安定化の ON/OFF。

6) 例外・フェイルソフト
   - ダイジェスト生成や LRU 操作に失敗した場合は静かにバイパス（従来経路）。
   - mode 安定化で明確に不適切（極端な入力変化）の場合は最新判定を優先。

DoD（受け入れ条件）
- 同一フレーム条件での再計算がキャッシュでスキップされ、CPU 使用率が有意に低下。
- 量子化による偽ヒットでの出力破綻なし（許容誤差内）。
- planar/projection の閾値近傍で出力が安定（揺れの視覚的消失）。
- 例外時のフェイルソフト（従来動作）を担保。

実装ステップ（チェックリスト）

1) ダイジェスト量子化ユーティリティ
- [x] `effects/clip.py` に量子化付きダイジェスト関数を追加（coords/offsets→量子化→blake2b-128）。
- [x] env/設定から `PXD_CLIP_DIGEST_STEP` を取得。

2) 結果 LRU の導入
- [x] `OrderedDict` ベースの LRU を追加（maxsize/env）。
- [x] キー整備: `(mode, mask_digest_q, target_digest_q, flags)`。
- [x] `max_verts` ガードを実装（env）。

3) モード安定化
- [x] `mask_digest`→`mode` の小さな辞書を導入。`PXD_CLIP_MODE_STABLE=1` のとき使用。
- [x] 判定が閾値近傍で揺れる場合は最後の mode を再使用（弱ピン止め）。

4) `clip()` への統合
- [x] planar/projection 分岐直後で結果 LRU ヒット確認→早期 return。
- [x] ミス時は通常計算→`max_verts` 以内なら LRU へ格納。

5) テスト/検証
- [ ] 同一入力フレームでのスキップ（計測/ログ）を確認。
- [ ] 量子化ステップの有無で結果差が目視許容内であること。
- [ ] mode 安定化 ON/OFF で閾値付近の揺れ方が変化すること。

備考
- マルチプロセス時は各プロセスで独立 LRU（現行設計と整合）。
- 量子化ステップは既存のパラメータ量子化（`PXD_PIPELINE_QUANT_STEP`）と別枠で管理し、過度なステップにしない（1e-4 推奨）。
