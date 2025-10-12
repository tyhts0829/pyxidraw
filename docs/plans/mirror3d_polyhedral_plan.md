# effect: mirror3d 拡張（多面体対称 T/O/I）実装計画

本ドキュメントは、mirror3d に「多面体対称（T=正四, O=正八/立方, I=正二十/正十二）」を導入するための詳細計画。既存の方位くさび（azimuth）モードに加えて `polyhedral` モードを追加し、群対称に基づく 3D 放射状ミラーを提供する。

## 目的と範囲（v1）
- 目的: 球面上の基本領域（球面三角形）を用い、有限反射群 A3/B3/H3（= T/O/I）により全空間へ鏡映/回転複製する。
- 範囲: T/O/I の3群に限定し、基本領域は大円境界（平面）で定義。回転のみ（偶置換）/反射込みの切り替えを提供。
- 互換: 既存 mirror3d の `azimuth` 機能は維持。`mode='polyhedral'` 選択時のみ本機能を有効化。

## API 案（mirror3d にモード追加）
- 関数: `mirror3d(g, *, mode: Literal['azimuth','polyhedral']='azimuth', group: Literal['T','O','I']|None=None, use_reflection: bool=False, orientation: tuple[float,float,float]|None=None, axis: tuple[float,float,float]=(0,0,1), roll_deg: float=0.0, cx: float=0.0, cy: float=0.0, cz: float=0.0, ...) -> Geometry`
  - `mode`: 'azimuth'（既存）/ 'polyhedral'（本拡張）。
  - `group`: 'T'|'O'|'I'（polyhedral 時に必須）。
  - `use_reflection`: True で反射を含む群（反射群）を使用（群の位数: A3=24, B3=48, H3=120）。False で回転部分群のみ（12/24/60）。
  - `orientation`: Euler 角 (rx,ry,rz) [rad]。指定時は `axis/roll_deg` より優先。省略時は `axis` を基準に `roll_deg` で回す簡易姿勢。
  - `axis/roll_deg`: 基準軸の方向と軸回り回転（orientation 未指定時の姿勢決定に使用）。
  - `cx,cy,cz`: 中心座標（反射/回転の pivot）。
  - 既存パラメータ（n_azimuth, phi0_deg, mirror_equator 等）は mode='polyhedral' では無効化（相互排他）。

`__param_meta__` 追加（案）:

```python
mirror3d.__param_meta__.update({
    'mode': {'choices': ['azimuth','polyhedral']},
    'group': {'choices': ['T','O','I']},
    'use_reflection': {'type': 'bool'},
    'roll_deg': {'min': -180.0, 'max': 180.0, 'step': 1.0},
    # orientation は GUI 未対応なら省略（コード/API で受け入れ）。
})
```

## 数学と設計
- 基本領域（球面三角形）: 3 つの大円（平面）で囲む。各境界は中心 `c` を通る鏡映平面。
  - A3/B3/H3 のコクセター行列 `m_ij` に対し、2 平面の成す角は `π/m_ij`。A3: (π/3, π/3, π/2), B3: (π/3, π/4, π/2), H3: (π/3, π/5, π/2)。
  - 実装では各群の「正規化済み法線ベクトル n1,n2,n3」を定数として持ち、姿勢（orientation/axis+roll）で回す。
- 反射行列: `R(n) = I - 2 n n^T`（n は単位長）。
- 群生成（有限反射群の近傍閉包）:
  - 生成元を {R(n1), R(n2), R(n3)} とし、BFS で閉包を構成。行列の同一性は `np.allclose`（EPS）で判定。
  - `use_reflection=False` の場合は `det(M) > 0` のものだけ（回転部分群）を採用。
  - 既知の位数（A3=24/12, B3=48/24, H3=120/60）に一致することをテストで保証。
- ソース抽出（クリップ）:
  - 3 半空間の AND: `s_k = n_k·(p-c) ≥ -EPS` を 3 つ満たす部分線のみ採用（INCLUDE_BOUNDARY=True 固定）。
- 複製:
  - すべての群要素 M に対し `p' = c + M (p - c)` を適用して複製。
  - 重複は EPS 量子化ハッシュで除去。

## 実装分割（内部関数）
- 法線/姿勢
  - `_poly_planes_T()/_O()/_I() -> tuple[np.ndarray, np.ndarray, np.ndarray]`（正規姿勢の法線 n1,n2,n3）
  - `_apply_orientation(normals, euler_or_axis_roll) -> tuple[n1',n2',n3']`
- 群生成
  - `_reflect_matrix(n: np.ndarray) -> np.ndarray`（3×3）
  - `_generate_reflection_group(n1,n2,n3, *, use_reflection: bool) -> list[np.ndarray]`
  - `_matrix_unique_key(M: np.ndarray) -> tuple`（EPS 丸め鍵）
- クリップ/複製
  - `_clip_polyline_halfspace_3d(vertices, normal, center) -> list[np.ndarray]`（既存流用）
  - `_clip_polyhedron_triangle(vertices, normals, center) -> list[np.ndarray]`（3 半空間 AND）
  - `_apply_transform_lines(lines, M, center) -> list[np.ndarray]`
  - `_dedup_lines(lines) -> list[np.ndarray]`（既存流用）
- キャッシュ
  - `_get_group_cache_key(group, use_reflection, orientation_hash) -> tuple`
  - `_GROUP_CACHE: dict[key, list[np.ndarray]]`（小規模 LRU でも可）

## TDD チェックリスト（段階実装）

フェーズA: スキャフォールド
- [x] A-1: mode パラメータを追加（polyhedral を受け付け、v1は 'T' 回転群に対応）。
- [x] A-2: `tests/effects/test_mirror3d_polyhedral_api.py` で最小テスト（T 回転群で12本）。

フェーズB: 法線と姿勢
- [ ] B-1: `_poly_planes_T/O/I` の単体（ペア角が π/m_ij に一致するか、cos の一致で検証）。
- [ ] B-2: `_apply_orientation` の単体（Euler と axis+roll の双方）。

フェーズC: 群生成
- [x] C-1: `_reflect_matrix` の単体（反射の性質: R^2=I, det=-1）。
- [x] C-2: `T` 回転群（12 元）を構成（頂点4軸の±120°回転 + 座標3軸の180°回転）。
- [x] C-3: O=48/24, I=120/60 を構成（O: 座標軸 90/270/180＋体対角 120/240＋辺軸 180。I: 5回/3回/2回軸の列挙）。

- [x] D-1: 3 半空間 AND によるクリップの単体（境界/交差の生成）。
- [x] D-2: mode='polyhedral' で group=T, use_reflection=False の最小 end-to-end（単点→12 本）。

フェーズE: 複製と重複除去
- [x] E-1: 回転のみで 12 本（T 群、単点入力）を検証。O は 24 本を検証。
- [x] E-2: 境界上の重複除去（azimuth の equator 反転で 4n→2n を確認済）。

フェーズF: 統合/ドキュメント
- [ ] F-1: mirror3d に polyhedral モードを結線（相互排他パラメータの検証を追加）。
- [ ] F-2: `tools/gen_g_stubs` 同期（mode/group/use_reflection/roll_deg を追補）。
- [ ] F-3: `architecture.md` に多面体対称の概要（群位数/基本領域/反射行列）を追記。

## テスト計画（要約）
- API: mode 切替の挙動、相互排他パラメータのバリデーション。
- 行列性質: 反射行列の性質、生成群の要素数（A3/B3/H3）。
- クリップ: 球面三角形の AND でのクリップ精度。
- 複製: 単点・単線に対する複製個数、重複除去。
- 姿勢/中心: orientation/axis+roll/c を変えた場合の等価性・平行移動の反映。
- パフォーマンス: 群行列のキャッシュ有無の切替、実行時間の目安（n_lines×|G| 程度）。

## 実装メモ
- 法線定数はコード内にハードコード（コメントでコクセター角を明記）。orientation で一括回転して利用。
- 群生成は BFS（反射3つの適用で十分）。要素の同一性は EPS 丸めで判定（直交性は allclose に頼る）。
- 回転部分群は `det(M)>0` でフィルタ。行列は 3×3 の float32 で揃える。
- 入出力は mirror3d の既存規約に準拠（EPS=1e-6, INCLUDE_BOUNDARY=True 固定）。

## オープン項目
- I 群（H3）の法線定数の妥当性確認（実装時に数表を用いて検証）。
- orientation 入力の GUI 露出（Euler か axis+roll）。
- 併用モード（azimuth × polyhedral）の意味付け（v1 では排他）。
