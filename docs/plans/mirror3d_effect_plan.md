# effect: mirror3d（真の3D放射状・球面くさび）実装計画

本ドキュメントは、3D 空間における球面くさび（大円境界）を用いた放射状ミラー `mirror3d` を独立エフェクトとして実装するための詳細計画。段階的に TDD で進めるため、小さな着手単位へ分解したチェックリストを提示する。

## 目的と範囲
- 目的: 中心 `(cx, cy, cz)` を通る複数の鏡映平面（大円）で構成される「球面くさび」を基準とし、ソース領域のみから3D空間へ放射状に鏡映複製する。
- 範囲（v1）:
  - 方位（azimuth）方向の n 等分（垂直面群、軸回りに Δφ=π/n）で 2n 個のくさびを構成。
  - オプションで「赤道面」（軸に垂直な 1 平面）で上下を反転し、さらに 2 倍（合計 4n）。
  - 中心移動 `c=(cx,cy,cz)` と軸方向の任意設定（既定は世界 Z 軸）。
  - 3D クリップ（半空間クリップ）と平面反射のみ（境界は include=固定・EPS=固定）。
- 範囲外（v2 以降に検討）:
  - 方位以外の第2平面族（傾斜面の等分）による更なる分割。
  - 多面体対称（正四/正八/正二十面体）等の群対称一括プリセット。
  - 球面座標の等面積分割などの高度な分布制御。

## API/パラメータ（案、v1）
- ファイル: `src/effects/mirror3d.py`
- 登録名: `mirror3d`（`@effect(name="mirror3d")`）
- 関数: `mirror3d(g: Geometry, *, n_azimuth: int = 1, cx: float = 0.0, cy: float = 0.0, cz: float = 0.0, axis: tuple[float, float, float] = (0.0, 0.0, 1.0), phi0_deg: float = 0.0, mirror_equator: bool = False, source_side: bool | tuple[bool, ...] = True) -> Geometry`
  - `n_azimuth`: 方位の等分数（min=1, max=64, step=1）。くさび角は Δφ=π/n_azimuth。
  - `cx,cy,cz`: 中心座標。
  - `axis`: くさびを並べる回転軸（単位化して使用）。既定は世界 Z 軸。
  - `phi0_deg`: くさびの開始角（方位の原点）。
  - `mirror_equator`: 軸に垂直な平面（赤道面）での反転を追加（true で 2 倍化）。
  - `source_side`: 基準くさびのソース側（半空間符号）を bool 指定（True=正側）。将来、境界面ごとの複数指定を許容（長さ不足は循環）。
  - 許容誤差と境界は固定: `EPS=1e-6`, `INCLUDE_BOUNDARY=True`（mirror.py と同様）。
- GUI/量子化: `__param_meta__` に min/max/step を付与（float は step で量子化）。

`__param_meta__`（v1 案）

```python
mirror3d.__param_meta__ = {
    'n_azimuth': {'min': 1, 'max': 64, 'step': 1},
    'cx': {'min': -10000.0, 'max': 10000.0, 'step': 0.1},
    'cy': {'min': -10000.0, 'max': 10000.0, 'step': 0.1},
    'cz': {'min': -10000.0, 'max': 10000.0, 'step': 0.1},
    'phi0_deg': {'min': -180.0, 'max': 180.0, 'step': 1.0},
}
```

## コア設計（v1）
- 境界面（大円）の定義
  - 方位くさびの境界は「軸を含む平面」2 枚（角間隔 Δφ=π/n_azimuth）。
  - 平面は中心を通るので、平面法線 `n` として「境界線方向に直交し軸に対して水平なベクトル」を採用。
  - `phi0_deg` で境界面の回転位相を決定（axis 回りに回す）。
- クリップ（ソース抽出）
  - 2 枚の境界面に対する半空間の AND を満たす部分だけを抽出。
  - `mirror_equator=True` の場合、赤道面（法線=axis）に対する半空間も AND へ追加（例: 上半空間）。
- 複製（鏡映複製）
  - まず非反転/境界1反転を準備（2 個）。
  - これらを `axis` 回りに `2π/n_azimuth` ステップで n 回回転（合計 2n 個）。
  - `mirror_equator=True` のとき、赤道反転を適用して再度 n 回回転し、さらに 2n 個（合計 4n）。
- 数値安定
  - `EPS=1e-6`, `INCLUDE_BOUNDARY=True` 固定。クリップ・交点・重複除去に使用。
  - 重複は頂点量子化ハッシュで除去。

## 実装分割（内部関数）
- ベクトル/行列ユーティリティ
  - `unit(v: np.ndarray) -> np.ndarray`
  - `rotate_around_axis(points: np.ndarray, axis: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray`
  - `reflect_across_plane(points: np.ndarray, normal: np.ndarray, center: np.ndarray) -> np.ndarray`
- クリップ系
  - `clip_polyline_halfspace_3d(vertices: np.ndarray, normal: np.ndarray, center: np.ndarray, side_sign: int) -> list[np.ndarray]`
  - `clip_polyline_wedge(vertices: np.ndarray, n0: np.ndarray, n1: np.ndarray, center: np.ndarray, side0: int, side1: int) -> list[np.ndarray]`
- 境界面生成
  - `compute_azimuth_plane_normals(n_azimuth: int, axis: np.ndarray, phi0: float) -> tuple[np.ndarray, np.ndarray]`（基本2枚）
  - `equator_normal(axis: np.ndarray) -> np.ndarray`
- 複製生成
  - `replicate_azimuth(src_lines: list[np.ndarray], axis: np.ndarray, step: float, center: np.ndarray) -> list[np.ndarray]`
  - `dedup_lines(lines: Iterable[np.ndarray]) -> list[np.ndarray]`

## TDD チェックリスト（小さな段階で前進）

フェーズA: スキャフォールド
- [x] A-1: `src/effects/mirror3d.py` の最小スケルトン（v1 実装）を作成し、`effects/__init__.py` へ import 追加。
- [x] A-2: `tests/effects/test_mirror3d_basic.py` を新規作成（インポート/関数呼び出しの最小確認）。

フェーズB: 基本幾何（単平面）
- [x] B-1: 3D 平面反射 `reflect_across_plane` の単体テスト（原点/任意中心、法線正規化の有無）。
- [x] B-2: 3D 半空間クリップ `clip_polyline_halfspace_3d` の単体テスト（内→外/外→内/境界/平行）。
- [x] B-3: 軸回り回転 `rotate_around_axis` の単体テスト（既知角: 90°, 180°）。

フェーズC: くさび生成とソース抽出
- [x] C-1: `compute_azimuth_plane_normals` のテスト（n=1/2/3、phi0 が境界法線に反映）。
- [x] C-2: `clip_polyline_wedge` のテスト（Z 軸基準で簡単な線がくさびにクリップされる）。
- [x] C-3: `equator_normal` のテスト（axis と一致）。

- フェーズD: 2n 複製（回転 + 反転）
- [x] D-1: n=1（半空間）で片側のみをソースに、反対側へ反射（2 本）。
- [x] D-2: n=3、source 内の 1 点から 2n=6 個の点が等角度 φ で出現（既存 mirror の n=3 テスト相当の 3D 版）。
- [x] D-3: `mirror_equator=True` で上下反転を加え、4n 個に増える。

フェーズE: 効果関数の統合
- [x] E-1: `mirror3d` 本体に v1 機能を結線（ソース抽出→複製→重複除去）。
- [x] E-2: 中心 `(cx,cy,cz)` と `axis` が反映されることをテスト。
- [x] E-3: `__param_meta__` を追加し、スタブ生成（`tools/gen_g_stubs`）の同期テストを緑化。

フェーズF: 安定化とドキュメント
- [x] F-1: 大きめ入力（グリッド/スパース点）での実用テスト（重複除去/実行時間）。
- [x] F-2: `architecture.md` に mirror3d 概要（平面式/複製則/境界方針）を追記。
- [x] F-3: 本計画（本ファイル）の完了ステータス更新。

## テスト計画（詳細）
- 基本反射/クリップ
  - 点/線/線分が単一平面で正しく反射・クリップされる。
  - 境界上（EPS 以内）の点は 1 回のみ保持。
- n_azimuth の検証
  - n=1: 基準面に対する 2 方向（±）のみ。
  - n=3: 2n=6 の等分角（φ=±φ0 + k·120°）。
- 赤道反転
  - mirror_equator=True で上下に等価な反射が生成される（Z 反転の確認）。
- 軸/中心
  - 任意 `axis`（例: (1,1,1) を正規化）と `c` が結果に反映。
- 重複/境界
  - 重複除去の確認（EPS 量子化ハッシュ）。

## 実装メモ
- 平面反射: `p' = c + (I - 2 nn^T)·(p - c)`（n は単位法線）。
- 半空間判定: `s = n·(p - c)`、`s >= -EPS`（INCLUDE=真）で内側。
- 交点: 3D 線分 a→b と平面 n·(x-c)=0 の交点は `t = sA/(sA-sB)` で線形補間。
- くさび境界法線: `axis` と境界方位ベクトル `u(φ0)` に直交な `n = unit(axis × u)`。
- 回転: 任意軸回転（ロドリゲスの回転公式）。

## 非目標（v1）
- 大円以外の境界（円錐/球面座標の等角度面）。
- 群対称プリセット（T/O/I）・等面積分割・高機能 UI。
- 2D mirror の API に mirror3d のパラメータを混在させること。

## オープン項目
- `axis` の GUI 指定（2 角度指定 or 3 成分入力）の UX。
- `source_side` の多境界拡張（将来の第2平面族導入時）。
- n_azimuth の実用上限（重複・実時間）と UI 制限。
