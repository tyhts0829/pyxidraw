# extrude エフェクト最適化（Numba/njit 検討と実施チェックリスト）

対象: `src/effects/extrude.py`

目的: 出力同等性を保ちつつ、押し出し（複製＋スケール＋側面接続）処理の Python ループと小配列生成を削減し、Numba による nopython カーネルへ集約してスループットを向上させる。

---

現状サマリ（2025-10 時点）
- 処理内容:
  - 各ポリラインを `subdivisions` 回細分化（頂点間の中点を挿入）。
  - `direction` を正規化し `distance` を掛けた `extrude_vec` で平行移動した複製線を生成。
  - `center_mode` に応じて複製線へスケールを適用（`auto` は重心基準、`origin` は原点基準）。
  - 元線・複製線に加えて、対応頂点を結ぶ 2 点線分を側面エッジとして出力（退化はスキップ）。
- 実装特性（ボトルネック候補）:
  - 細分化は Python ループと list→`np.asarray` で多量の小配列を生成。
  - 出力は `out_lines: list[np.ndarray]` にラグドに積み、最後に `np.vstack`。2 点線分も都度 `np.asarray`。
  - 退化判定は `np.allclose(..., atol=1e-8)` を各頂点で評価。
  - Numba は未使用。

---

最適化方針（Numba を“効くところ”へ集中）
- 2 パス方式（count→allocate→fill）で前方確保し、出力配列に直接書き込む。
- 細分化・スケール・側面生成までを nopython カーネル内で完結。
- 可変長出力は offsets で管理し、list of arrays を廃止。
- 数値型は `float32`/`int32` を基本とし、変換回数を削減。

---

提案 API/カーネル設計（スケッチ）

- 入口（Python 側ラッパ）
  - `_extrude_dispatch(coords: float32[:, :], offsets: int32[:], direction: float32[:], distance: float32, scale: float32, subdivisions: int32, center_mode: int32) -> (float32[:, :], int32[:])`
    - 環境変数で numba 経路を選択（`PYX_USE_NUMBA`, `PYX_USE_NUMBA_EXTRUDE`, 互換 `PXD_USE_NUMBA_EXTRUDE`）。
    - 非 numba 環境では現行ロジック（もしくは numba と同一 2 パス形の純 Python 実装）にフォールバック。

- カウント/フィル
  - `@njit(cache=True)`  # fastmath は無効（数値順序の差異を避ける）
    - `subdivide_len(n: int32, s: int32) -> int32`
      - `return (1 << s) * (n - 1) + 1` で最終頂点数を即時算出。
    - `subdivide_line_njit(v: float32[:, :], s: int32, out: float32[:, :]) -> int32`
      - 2^s 回の中点挿入をインデックスで構成。最終配列 `out` に書き込み（長さを返す）。
    - `extrude_count_line(v: float32[:, :], extrude_vec: float32[:], scale: float32, center_mode: int32, eps: float32) -> (int32 /*m*/, int32 /*edge_cnt*/)`
      - m: 細分後の頂点数。edge_cnt: 非退化側面数。
      - `auto` は `centroid_v = mean(v); centroid = centroid_v + extrude_vec` を利用（`extruded_base` を明示生成しない）。
      - `origin` は `extruded_j = scale * v[j] + scale * extrude_vec`。
      - 退化判定は現行の `np.allclose(a, b, atol=1e-8)` と同等（`rtol=1e-5, atol=1e-8`）とする。
        - numba 内では `abs(a_i - b_i) <= (atol + rtol * abs(b_i))` を i=0..2 で満たすかの論理積で実装。
    - `extrude_fill_line(v, extrude_vec, scale, center_mode, eps, out_c, out_o, vc, oc) -> (vc, oc)`
      - out_c: `float32[:, :]` 連結頂点バッファ、out_o: `int32[:]` offsets。
      - 1) 元線 m 点、2) 複製線 m 点、3) 非退化側面 E 本 × 2 点を順に書き込み、オフセットを更新。
    - `extrude_kernel(coords, offsets, direction, distance, scale, subdivisions, center_mode, eps) -> (new_coords, new_offsets)`
      - 1 パス目で各ラインごとの (m, E) を算出→総頂点数 `sum(2*m + 2*E)` と出力ポリライン数 `sum(2 + E)` を確定。
      - 2 パス目で `subdivide_line_njit`→`extrude_fill_line` を行い、配列に直書き。

- パラメータ/型
  - `direction` は `float32[3]`、`distance` は `float32`。`norm < 1e-9 or distance == 0` で `extrude_vec = 0`。
  - `eps` は現行 `atol=1e-8` に合わせ `1e-8`（`float32` 基準で可）。
  - `center_mode`: `0='origin'`, `1='auto'` の int フラグ。

---

Python レイヤの改修
- `extrude(...)` 内での分岐:
  - `coords, offsets = g.as_arrays(copy=False)` は現行維持。
  - クランプ/型変換（`MAX_*`）は Python 側で実施。
  - `direction_vec` の正規化→`extrude_vec` 算出は Python 側で行い、`float32[3]` を numba に渡す。
  - 出力順序は「元線 → 複製線 → 側面エッジ（j 昇順）」を厳密維持（offsets も同順で更新）。
  - numba 経路が有効なら `_extrude_numba(...)` へ委譲。無効時は現行実装 or 同等の 2 パス純 Python 実装へ。
- 返り値は `Geometry(new_coords, new_offsets)`（現行踏襲）。dtype は `float32`/`int32`。

---

期待効果（目安）
- 細分化のループ・小配列生成の排除により、分割回数が大きい場合に 1.5〜3.0×。
- 側面生成（頂点ごと 2 点線分）を配列直書きすることで、リスト操作主体の現行より 1.3〜2.0×。
- 入力ライン数・頂点数が多いほど効果が大きい見込み。

注意/トレードオフ
- 初回 JIT コストがあるため、非常に小さな入力では体感差が小さい可能性。
- `float32` 固定により、極端なスケール・距離での丸め挙動差が理論上あり得るが、現行も `float32` で計算しており互換。
- 退化判定は `np.allclose(a, b, rtol=1e-5, atol=1e-8)` と等価に実装（現行と同等判定）。

---

検証方針（DoD）
- 機能同等性
  - 代表入力で旧実装と出力 polylines の本数・頂点列が一致（許容誤差: 1e-6 以内）。
  - 空/2 頂点/ゼロ距離/ゼロ方向、`subdivisions=0`/大値、`center_mode` 各種の境界で安定。
- 性能
  - 細分回数 4〜8、頂点数 1e4 規模の入力で 1.5× 以上の短縮を目安。
- チェック
  - 変更ファイルに対する `ruff/black/isort/mypy/pytest -q` 緑。
  - 公開 API 影響なし（`__param_meta__`/docstring はそのまま）。

---

実施チェックリスト（段階導入）
- [ ] ベースライン計測（代表ケース: subdivisions 多/少、center_mode 別）
- [ ] 2 パス構造の純 Python 版（同一 API）を用意し、現行実装と一致性テスト
- [ ] `subdivide_len`/`subdivide_line_njit` の実装とテスト
- [ ] `extrude_count_line`/`extrude_fill_line` の実装とテスト
- [ ] `extrude_kernel` の実装（全ライン一括）
- [ ] Python ラッパ `_extrude_dispatch` と環境スイッチ（`PYX_USE_NUMBA[_EXTRUDE]`）
- [ ] ベンチと回帰テスト（機能/性能）
- [ ] 必要に応じて `architecture.md` に補足（2 パス/offsets 管理への移行）

---

実装メモ（補足）
- 重心は `sum(v, axis=0) / m` を 1 ループで算出。`auto` の重心は `centroid_v + extrude_vec` で良い。
- 退化判定（numba 内）は `abs(a_i - b_i) <= (1e-8 + 1e-5 * abs(b_i))` を i=0..2 で満たすかで代用（`np.allclose` 相当）。
- `subdivide_line_njit` は `for it in range(s):` で in-place 的に 2 段バッファを切替えるか、最終配列へ段階的に書き込む。単純さ優先で後者を推奨。

---

次アクション
- このチェックリストの方針で実装に進めてよいか確認してください。承認後、純 Python 2 パス → numba カーネルの順に小さく分割して反映します。
