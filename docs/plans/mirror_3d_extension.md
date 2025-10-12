# mirror 3D 拡張検討（cz 対応）

本ドキュメントは、既存の `mirror` effect を 3 次元（Z 平面も含む）に拡張する可否・設計案・影響範囲を整理する検討ノート。

## 目的
- 現状の `mirror` は XY 内での対称（x=cx, y=cy）に限定。`cz` を導入し、`z=cz` による鏡映も扱いたい。
- 3 軸の直交平面（x, y, z）に対して、半空間/象限/八分体（octant）ベースでソース領域を選び、他領域へ複製する 3D ミラーを提供する。

## 現状（要約）
- 実装: `src/effects/mirror.py`
- パラメータ: `n_mirror`（2D 放射状: 1/2/≥3）、`cx, cy`、`source_side: bool | Sequence[bool]`（x/y 平面用）。
- クリップ: 軸整列半空間（x, y）に対する線分クリップ（`_clip_polyline_halfspace`）。
- 生成: n=1 は x ミラー、n=2 は x/y/x+y、n≥3 は 2D の扇形複製（回転 n 個 + 反転後回転 n 個）。Z は不変。
- 数値: `EPS=1e-6`、`INCLUDE_BOUNDARY=True` をモジュール定数として固定。

## 要件整理（3D 拡張）
- 新たに `cz: float` を導入し、`z=cz` での鏡映を可能にする。
- ソース領域の選択は「半空間のAND」で定義し、以下の3通りを包含:
  - 1 平面（半空間）: x または y または z のいずれか一つ
  - 2 平面（象限相当）: 任意の 2 軸（例: x と z）
  - 3 平面（八分体）: x, y, z の全て
- ソース側の指定は軸ごとに `True/False`（True=正側、False=負側）。未指定軸はソース制約に含めない（＝制約無し）。
- Z を含むミラーでは、生成は選択した軸の反転組合せで 2^k 個（k=選択軸数）を複製（重複は除去）。
- 既存の 2D 放射状ミラー（`n_mirror>=3`）は今回の拡張の対象外（互換維持）。

## 設計オプション
- 案 A（推奨）: 直交 3 平面モード（orthogonal）を追加
  - `mode: Literal['radial2d','orthogonal']` を導入。既定は互換のため `radial2d`。
  - `orthogonal` のとき `axes: tuple[bool|None, bool|None, bool|None]`（None=未選択）または `use_axes: tuple[bool, bool, bool]` と `source_side: tuple[bool|None, bool|None, bool|None]` を解釈。
  - クリップは選択軸の半空間を順に適用（x→y→z）。
  - 生成は選択軸について反転ビット列の全組合せで複製（2^k）。
- 案 B: 既存 API に最小追加（互換性重視）
  - 新規引数 `cz: float = 0.0` を追加。
  - 新規引数 `use_z: bool = False`（Z 平面を選択）を追加。
  - `source_side` は `(sx, sy, sz)` として長さ 3 を許容（不足は循環）。
  - `n_mirror` はそのまま（1/2/≥3 は 2D のみ）。`use_z=True` の場合は 2D 分岐をスキップし、直交 1～3 軸のロジックを用いる。
- 案 C: 完全統一インターフェース（将来）
  - 2D 放射状（角楔）と 3D 直交（半空間）をモードで完全分離し、各モードのパラメータを明確化。

結論: v1.1 としては「案 B（最小追加）」を採用し、将来 `mode` 導入で案 C に寄せるのが安全。

## API 変更案（案 B）
- 追加: `cz: float = 0.0`
- 追加: `use_z: bool | None = None`
  - `None`（既定）: 互換モード（従来通り 2D の n_mirror 分岐）。
  - `True`/`False`: Z 平面を（使用/不使用）として直交モードを有効化。`True` の場合、x/y の扱いは `n_mirror` ではなく `axes` 選択で制御。
- 追加（任意）: `axes: tuple[bool, bool, bool] | None`（例: `(True, False, True)` で x と z を選択）。`None` 時は `n_mirror` と `use_z` から推定。
- 既存: `source_side: bool | Sequence[bool]`
  - 長さ 3 を許容（不足は循環適用）。順序は `[sx, sy, sz]`。
- 量子化/境界/誤差: 既存の `EPS=1e-6`, `INCLUDE_BOUNDARY=True` を流用（固定）。

`__param_meta__`（例）

```python
mirror.__param_meta__.update({
    'cz': {'min': -10000.0, 'max': 10000.0, 'step': 0.1},
    # GUI では `axes`/`use_z` をトグルで露出（詳細仕様は UI 層）。
})
```

## コアアルゴリズム（直交モード）
1) ソース抽出（クリップ）
   - 選択軸それぞれの半空間で `_clip_polyline_halfspace(axis=0/1/2, thresh=c*, side=±1)` を順次適用。
   - Z については `axis=2, thresh=cz` を使用。`side` は `source_side` から導出（True=+1/False=-1）。
2) 複製（鏡映）
   - 選択された各軸について反転の有無の全組合せを列挙（ビット列 0..(2^k-1)）。
   - 各ビット集合に対して `_reflect_x/_reflect_y/_reflect_z` を順に適用。
3) 正規化/重複除去
   - 既存の量子化ハッシュ（EPS）で重複ポリラインを除去。

擬似コード（反転組合せ）

```python
axes = [(axis, thresh) for axis, use in [(0,cx),(1,cy),(2,cz)] if use]
K = len(axes)
for p in src_lines:
    for mask in range(1 << K):
        q = p
        for i, (axis, c) in enumerate(axes):
            if (mask >> i) & 1:
                q = reflect_axis(q, axis, c)
        out_lines.append(q)
```

## テスト計画（追加）
- Z 半空間の基本
  - `cz=0`、`source_side_z=True` で z>=0 の点/線のみをソース→ z<0 へ鏡映。下側に元からある線は削除。
- XZ 象限（2 軸）
  - x>=cx かつ z>=cz のみソース→残り 3 象限へ鏡映。元から他象限の線は削除。
- XYZ 八分体（3 軸）
  - (x>=cx, y>=cy, z>=cz) のみソース→他 7 八分体へ鏡映。元から他八分体の線は削除。
- 交差クリップ
  - z 軸方向に跨ぐ線が正しくクリップされる（交点が z=cz 上に生成）。
- 境界/重複
  - 境界上の線は 1 回のみ出現。重複除去が働く。
- Z 不変性の撤回
  - 2D モード（従来）は Z 不変。直交モードでは Z 反転あり（z' = 2*cz - z）。

## エッジケース/検討事項
- `source_side` の長さが 1/2/3 の混在: 不足は循環適用（既存仕様を踏襲）。
- `axes` を指定しない場合の推論規則: `use_z` と `n_mirror` から x/y/z を推論（例: n=1→x のみ、n=2→x,y、n≥3 かつ use_z→x,y,z）。
- 大量複製: 選択軸 K に対し 2^K 個の複製を生成。K=3 で最大 8 倍。入力サイズと相談（DoS 回避）。
- 数値安定性: 既存 EPS で十分だが、z クリップ周りのしきい検討（極端に薄い線）。
- パイプラインキャッシュ: パラメータ追加により鍵（pipeline_key）が変わる。スタブ更新も必要。

## 互換性
- 既存呼び出しは非破壊。
  - `use_z`/`axes` を指定しない限り従来の 2D 振る舞い。
  - 新規 `cz` のデフォルトは 0.0（影響なし）。

## 性能影響（概算）
- クリップ: O(M·K)、K は選択軸数（≤3）。
- 複製: O(M·2^K)。K=3 の場合 8 倍。入力フィルタ（ソースクリップ）により抑制。
- メモリ: 出力頂点/線本数は複製係数に比例。重複除去で若干減少。

## 実装ステップ（チェックリスト）
- [ ] API/パラメータ追加（`cz`, `use_z`, `axes`）と `__param_meta__` 更新
- [ ] クリップ: z 平面対応（`axis=2, thresh=cz`）
- [ ] 複製: 反転組合せの一般化（x/y/z 任意）
- [ ] テスト: Z 半空間/象限/八分体/交差/境界/重複
- [ ] スタブ再生成（`tools/gen_g_stubs`）
- [ ] ドキュメント更新（`architecture.md` と本計画の更新）

## 結論
- 実装は十分可能。既存実装の x/y クリップと反射の一般化で 3D 対応を安全に拡張できる。
- v1.1 では「案 B（最小追加）」で導入し、将来的に `mode='orthogonal'` の明示導入を検討する。

