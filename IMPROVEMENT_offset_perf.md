# offset エフェクト高速化アイデア（チェックリスト）

目的
- `src/effects/offset.py` の `offset` 関数（src/effects/offset.py:36）の実行時間短縮。
- 大規模ポリライン／多数ライン入力時の待ち時間を体感で削減（×2〜×10 目標、段階導入）。

現状ボトルネックの想定
- Shapely `buffer()` 呼び出し（ジオメトリ変換の中核）
- ライン単位での逐次処理（Python ループ）
- 3D→XY 射影/復元の反復（ほぼ安価だが回数が多いと効く）
- 出力結合（`np.vstack`）による一時アロケーション

---

優先度A（Quick Win / 互換性維持）
- [ ] XY 平面ショートカット: 入力 z の分散が閾値以下なら `transform_to_xy_plane`/`transform_back` をスキップ（恒等復元）。
  - 期待効果: 2D 入力多数ケースで 1.2〜1.5x。
  - 実装ポイント: z 範囲 `max(z)-min(z) <= 1e-9` 等の閾値判定。
- [ ] `_scaling` のオプション化/デフォルト無効化（速度と一貫性のため実行回数を削減）。
  - 期待効果: 大量ラインで 1.05〜1.2x（後段の配列走査削減）。
  - 備考: 幾何的意味の変化を避ける観点でも推奨（機能改善票と整合）。
- [ ] `segments_per_circle` の自動下げ（距離が小さい/曲率が低い箇所で分割数を抑制）。
  - 期待効果: 距離が小さいケースで顕著（1.5〜3x）。
  - 方式案: `resolution = max(4, min(user_value, ceil(2 + sqrt(distance*scale))))` の保守的なルール。
- [ ] メモリ最適化: 出力配列の二段確保（長さ集計→`np.empty(total,3)`→コピー）で `np.vstack` の再アロケを回避。
  - 期待効果: ライン数・頂点数が多いほど有利（1.1〜1.3x）。

優先度B（並列・ベクトル化）
- [ ] 並列化（安全版: `ProcessPoolExecutor`）でラインを分割バッチ処理。
  - 条件: Shapely v1 でも CPU 並列可（プロセス分割）。Shapely v2 ならスレッドでも効果が出やすい。
  - 期待効果: コア数に概ね比例（I/Oなし前提で 2〜6x）。
  - 実装ポイント: チャンク分割、`join_style`/`resolution` は不変引数、結果は配列に集約。
- [ ] Shapely v2 のベクトル化API活用（可能なら）
  - 方式案: `MultiLineString` または配列化したジオメトリを一括 `buffer` → 個別抽出。
  - 注意: 一括 `buffer` はジオメトリが結合（dissolve）される可能性。要件に応じて不可。

優先度C（前処理の削減）
- [ ] 前段簡略化: Douglas-Peucker（RDP）で頂点数削減（`tolerance ≈ distance/4` など保守値）。
  - 期待効果: 複雑形状で 2〜10x（Shapely の入力点数減）。
  - 互換性: `preserve_topology=True` を前提に「外観上の差が僅少」な程度に制御。
- [ ] Shapely の `simplify` 利用（依存追加なし）か、独自の RDP 実装（numpy）を導入。
  - 方針: まず Shapely `simplify`、効果不足なら軽量RDPを内製。

優先度D（アルゴリズム置換・依存追加: Ask-first）
- [ ] Clipper ベースのオフセット（`pyclipper`）に切替またはオプション化。
  - 期待効果: 大規模ポリゴンで 5〜20x、安定性高。
  - 実装要点: mm→整数スケール（例: ×1000）、`CO` で `Miter/round/bevel` 指定、出力を mm に戻す。
  - リスク: 依存追加（Ask-first）。3D 射影/復元は従来通り前後に適用。
- [ ] 直交/斜交セグメントの幾何オフセットを NumPy で内製（丸めは円近似）。
  - 期待効果: 中〜大規模で 3〜10x（Union/自己交差解決の実装コストは高め）。
  - リスク: 仕様の角ケース（自己交差/鋭角/非単純形）対応の開発工数。

優先度E（I/O・フラグ設計）
- [ ] `assume_planar: bool=False` 追加（2D前提を許可）。有効時は射影/復元を完全スキップ。
- [ ] `fast_mode: bool=False` 追加（前処理簡略化+解像度自動ダウン+スケーリング無効化の束ね）
- [ ] `workers: int|None` 追加（None→自動、1→現状維持、>1→並列）
- [ ] `simplify_tol: float|None` 追加（明示指定時のみ簡略化）

計測・しきい値（導入順に検証）
- [ ] ミニベンチ（代表 3 ケース）を追加: 小/中/大（総頂点数、ライン数、平面性の有無）。
- [ ] 変更単位の高速チェック: `pytest -q -m perf -k offset`（新規マーカー）
- [ ] 回帰基準: 速度 1.5x 以上か、品質差が視覚/数値で許容内（頂点数差、形状差メトリクス）

実装順（提案）
1) Quick Win 3点（XYショートカット/スケーリング無効/解像度自動ダウン）
2) メモリ最適化（出力二段確保）
3) オプションフラグ（`assume_planar`/`fast_mode`）
4) 並列化（`workers`）
5) 簡略化（`simplify` or RDP）
6) Clipper 導入（Ask-first）

影響/互換性
- 既定値は互換を維持（挙動切替はフラグ/自動ダウンは保守的）。
- `fast_mode` 有効時は形状の滑らかさ/頂点数が減る可能性（明示 opt-in）。
- 依存追加（`pyclipper`）は Ask-first のうえオプション実装。

事前確認事項（要ご判断）
1. 品質と速度のトレードオフ: `fast_mode` を導入してよいか。既定は OFF で運用可？
2. 依存追加: `pyclipper` 採用を検討しますか？（採用時はON/OFF切替可能に）
3. 並列実行: デフォルト `workers=None`（自動）で導入してよいか。
4. 簡略化: まず `shapely.simplify(preserve_topology=True)` を軽微な既定として許容しますか？（閾値は距離に基づく保守値）

参考（具体的タッチポイント）
- `offset()` 本体: `src/effects/offset.py:36`
- Shapely 呼び出し部: `src/effects/offset.py:74` 付近（`LineString(...).buffer(...)`）
- 3D 射影/復元: `src/util/geom3d_ops.py:24`, `src/util/geom3d_ops.py:74`
- スケーリング: `src/effects/offset.py:120`（`_scaling` と呼び出し）

進捗（このファイルの役割）
- [x] 高速化アイデアのチェックリスト作成（本ファイル）
- [ ] 実装は未着手（確認後に着工）

