% `clip` 前段固定費の削減計画（結果キャッシュ命中時でも残る高負荷の対処）

目的
- 結果キャッシュ命中時でも毎フレーム実行される前段処理（O(N)）を軽量化し、CPU 使用率をさらに下げる。
- 対象: 1) マスク収集/閉路化コピー 2) 量子化ダイジェスト計算 3) 共平面判定/整列（choose_coplanar_frame）。
- ユーザー側のスケッチは変更不要。

現状の固定費（要約）
- マスクリング収集: `outline` から閉路リング配列を毎回作成（`vstack`/コピー）。
- ダイジェスト計算: `coords` を量子化（`rint`→`int64`）し blake2b でハッシュ（ターゲット/マスク両方）。
- 共平面判定: ターゲット+マスク連結→回転/場合により SVD で XY 整列（キャッシュ前に実行）。

実装アイデア（優先度順）

1) マスク収集の前処理キャッシュ（リング再構築の省略）
- 目的: `_collect_mask_rings_from_outline` の `ensure_closed`/`vstack`/コピーを避ける。
- 方針:
  - `outline` の内容ダイジェスト（量子化）をキーに、閉路済み `rings3d` を LRU 再利用。
  - 既存の「マスク前処理 LRU」（projection/planar 用）に「収集済み rings3d 生配列」も保存。
  - `outline` が Geometry の場合は WeakKeyDictionary による per-object メモ（id 固定時の再計算省略）も併用。
- 期待効果: 大きなリング/複数輪郭でのメモリ確保/コピーコストの削減。

2) ダイジェスト計算の軽量化
- 目的: `_quantized_digest_for_coords/_rings` の配列変換&ハッシュを安くする。
- 方針:
  - ストリーミング更新: 大配列を一定チャンクに分割して `h.update()`、ピークメモリを抑える（行単位/ブロック単位）。
  - 型/演算の簡素化: `float64→rint→int64` を `float32→round→int32` に切替（環境でオプトイン）。
  - サンプリングハッシュ（任意）: `PXD_CLIP_DIGEST_STRIDE=k` で k ストライドごとの頂点だけ量子化→ハッシュし、合わせて bbox/頂点数も加える（擬似重み）。
  - per-Geometry メモ: `WeakKeyDictionary[Geometry] -> (digest, meta)` による使い回し（coords/buffer アドレス/shape の一致を簡易検証）。
- 期待効果: 毎フレームの量子化/ハッシュの CPU/メモリ帯域削減。

3) 共平面判定のショートサーキット/スキップ
- 目的: `choose_coplanar_frame` のコストを避ける。
- 方針:
  - モード安定化連携: `MODE_PIN[mask_digest] == 'proj'` の場合は `choose_coplanar_frame` を丸ごとスキップ（現行は実行後に分岐）。
  - 早期 XY 判定: ターゲット/マスクの z スパン（min/max）を cheap に計算してしきい値以内なら整列不要（R=I, z=0）とみなす。
  - マスク基準整列: 共平面推定はターゲットを含めず「マスクだけ」に限定（安定化済みの姿勢を優先採用）。
- 期待効果: projection モード時の完全スキップ、planar モード時も cheap 判定で多くのケースを回避。

4) 結果キー生成の早期判定（整列不要）
- 目的: キャッシュヒット判定までの処理を最小化。
- 方針:
  - ダイジェスト→モード確定（ピン）→結果 LRU ヒット判定→return の順で、整列は「必要時のみ」実行。
  - 現状でも概ねそうなっているが、`choose_coplanar_frame` の呼び出し位置を「proj 固定のときは通らない」ように再配置。

5) 小粒な改善
- 収集時の閉路化: 既閉路検出の閾値/条件を厳格化して `vstack` の発生を減らす（完全一致/`np.allclose` を優先）。
- `np.asarray(..., copy=False)` の徹底と dtype 変換の最小化。
- `rings3d` → `coords2d/offsets` の前処理は projection LRU に保存済みのものを最優先で流用（`rings_xy` 個別配列の再構築を避ける）。

リスクと対策
- サンプリングハッシュ: 偽一致リスク → bbox/頂点数/総長さの付加でキーの衝突率を低減、既定はフルハッシュ。
- モード固定/スキップ: 閾値誤判定 → 最新判定が大きく変わる場合はピンを更新、常に安全側（planar へ）に倒すオプション。

環境スイッチ（提案）
- `PXD_CLIP_DIGEST_STRIDE`（未指定: 1=フル）
- `PXD_CLIP_USE_FLOAT32_DIGEST`（0/1, 既定 0）
- `PXD_CLIP_MODE_PIN_MASK_ONLY`（0/1, 既定 1）: マスクのみで mode を決める

DoD（受け入れ）
- 結果キャッシュ命中時に、前段処理の CPU 時間が更に低下（プロファイル/ログで確認）。
- 出力差はなし（誤判定/衝突による破綻なし）。
- 失敗/例外時は従来経路にフォールバック（フェイルソフト）。

実装ステップ（チェックリスト）

1) 収集キャッシュ
- [x] `outline` 内容 digest で `rings3d` を LRU 保存/再利用（収集済み生配列）。
- [ ] WeakKeyDictionary による per-Geometry digest/memo を併用。

2) ダイジェスト軽量化
- [x] チャンク更新/float32+int32 量子化に対応（env で切替）。
- [x] サンプリングハッシュ（stride）を実装（既定オフ）。
- [ ] per-Geometry digest のメモ化を導入。

3) 共平面判定スキップ
- [x] `MODE_PIN == 'proj'` の場合、`choose_coplanar_frame` をスキップ。
- [ ] z スパン cheap 判定で整列不要ケースを early return。
- [ ] マスク基準整列（必要時のみ target を含めない）。

4) 呼び出し順の整理
- [x] ダイジェスト→モード決定→結果 LRU 判定を整列前に確定するよう再配置（proj ピン時）。
