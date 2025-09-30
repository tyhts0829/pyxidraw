# dash エフェクト最適化（案2: while ベクトル化＋2パス前方確保）

対象: `src/effects/dash.py`

目的: 出力同等性（幾何的に等価 or 許容誤差内）を保ちつつ、破線生成の計算コストと Python ループ回数を削減する。

背景（現状）
- 各ポリラインで弧長 `s` を計算し、`while` でダッシュ開始/終了距離を進め、`_interpolate_segment` で部分線を切り出している。
- 1 ダッシュあたり `searchsorted` を2回呼ぶため、Python 側のループ/関数呼び出しが支配的になりがち。

---

方針（案2の確定）
- `while` は残しつつ「端点探索と補間」を配列化し、Python の関与を最小化する。
- さらに将来の `njit` を見据え、2パス構成（count/fill）で前方確保し、出力配列に直書きする。

アルゴリズム（1 本のポリライン）

```
segments = v[1:] - v[:-1]
dist = sqrt((segments**2).sum(axis=1))
s = r_[0, cumsum(dist)]  # 弧長
L = s[-1]
if L <= 0: return 原線

starts = arange(0.0, L, pattern)           # [0, pattern, 2*pattern, ...]
ends   = minimum(starts + dash, L)

# 索引を配列で一括取得
s_idx  = searchsorted(s, starts, side='left')
e_idx  = searchsorted(s, ends,   side='left')

# 端点を配列で線形補間（den は 0 回避）
s0 = maximum(s_idx - 1, 0)
e0 = maximum(e_idx - 1, 0)
den_s = maximum(s[s_idx] - s[s0], eps)
den_e = maximum(s[e_idx] - s[e0], eps)
ts = (starts - s[s0]) / den_s
te = (ends   - s[e0]) / den_e
start_pts = v[s0] + (v[s_idx] - v[s0]) * ts[:, None]
end_pts   = v[e0] + (v[e_idx] - v[e0]) * te[:, None]

# 各ダッシュ i について、
#   if s_idx[i] == e_idx[i]: [start_pt, end_pt]
#   else: [start_pt] + v[s_idx[i]:e_idx[i]] + [end_pt]
# を出力配列に書き出す。
```

計算量と期待効果
- `s` の計算 O(N)。索引探索は配列で2回（O(M log N) だが呼び出し回数は2回）。
- 端点補間は完全ベクトル化。ラグド結合のみ最小限の Python ループで処理。
- 既存より 1.3〜2.0× 程度の短縮を主に Python オーバーヘッド削減で期待。

数値安定化
- `eps = 1e-12`（float64 基準）を使用して 0 除算を回避。
- `side='left'` を基本とし、0 長区間/境界一致に頑健。
- 出力 dtype は `float32` に統一。

---

Numba/njit を見据えた設計
- 2 パス方式 + 前方確保（pre-allocation）で、ラグド構造の連結を避ける。
- コアは for ループ + `cumsum`/`searchsorted` + 手作り補間に限定（非対応 API を回避）。
- スケッチ

```
# count
def _dash_count_line(v: float32[:, :], dash: float32, gap: float32) -> (int32, int32):
    # returns: (dash_count, vertex_total)

# fill
def _dash_fill_line(v: float32[:, :], dash: float32, gap: float32,
                    out_c: float32[:, :], out_o: int32[:], vc: int32, oc: int32) -> (int32, int32):
    # writes dashed polylines; returns new cursors

# whole
def dash_kernel(coords: float32[:, :], offsets: int32[:], dash: float32, gap: float32)
        -> (float32[:, :], int32[:]):
    # 2 パスで配列確保→充填
```

- 実装時はまず純 Python で同構造を採用し、将来 `try-import numba` で `njit(cache=True)` を任意有効化。
- 既定は numba 無し（依存追加なし）。環境変数で opt-in（例: `PYX_USE_NUMBA=1` または個別に `PYX_USE_NUMBA_DASH=1`。レガシー `PXD_USE_NUMBA_DASH` も可）。

---

マイクロ最適化のヒント
- `np.sqrt((seg**2).sum(axis=1))` は十分高速。`np.linalg.norm` より軽量。
- 検索の side/境界処理を固定して条件分岐を減らす。
- `float32`/`int32` を通し、キャスト回数を最小化。

---

検証方針（DoD）
- 機能同等性
  - ランダム折れ線で旧版との破線区間の弧長誤差 ≤ 1e-5、端点位置誤差 ≤ 1e-6。
  - 空/単一点/ゼロ長/極短線での安定動作（原線保持）。
- 性能
  - 高密度ポリライン（N=1e5, M=数百）で 1.3〜2.0× 短縮を目標。
- テスト
  - `pytest -q -m smoke` 緑。主要 `tests/perf` の退行なし。

---

実施チェックリスト
- [ ] ベースライン計測（小規模）で現行性能を記録
- [x] 2 パス構造（count/fill）で `dash.py` を実装
- [ ] 数値安定化（eps/side）の固定化と境界テスト
- [ ] 回帰テスト（機能/性能）
- [ ] `architecture.md` 差分があれば更新
- [x]（任意）`PYX_USE_NUMBA[_DASH]` による njit 経路の用意（依存は追加しない）

---

次アクション
- この方針（案2＋2パス）で実装へ進めてよいか確認してください。
