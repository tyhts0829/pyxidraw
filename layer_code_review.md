# Layer 実装コードレビュー

## 指摘事項

1. **High – レイヤー名重複時の override キー不整合で GUI 値が効かない**

   - `src/engine/ui/parameters/manager.py:324-327,349-384` ではレイヤー名が重複しても `layer.<name>` のまま Descriptor を登録し、2 枚目以降は同じキーに上書きされる。
   - `src/engine/runtime/worker.py:160-204` では同じ名前に `_idx` サフィックスを付けたキーで override を引くため、GUI 側に `layer.<name>_1` などの Descriptor が存在せず 2 枚目以降に値が反映されない。
   - `L.of(..., name="foo")` のように同名レイヤーを返すと、GUI の色/太さ調整が先頭レイヤーだけに限定され、それ以外は元値のままになる。例外や警告が無いため気づきにくい。
   - 提案: GUI 側とワーカ側でキー生成を揃え、重複名を許容するなら同じキーを共有するか、両側ともに明示的なインデックス付きキーに統一する。
     → もし同名レイヤー名が指定された場合は、foo_1 みたいに suffix を自動で付与するようにして。

2. **Medium – LazyGeometry を 1 レイヤー描画中に二重 realize**

   - `src/engine/render/renderer.py:180` の `_upload_geometry()` が `LazyGeometry` を `realize()` した後、同じループで `renderer.py:189-195` でもスナップショット用に `realize()` を再実行している。
   - `LazyGeometry.realize()` はキャッシュ済みでもシグネチャ計算やキャッシュ参照が走るため、重い形状では毎レイヤーで余分なコストが掛かる。
   - 提案: `_upload_geometry()` の実体化結果を使い回すか、最初の `realize()` の結果をローカルに保持してスナップショットに流用する。
   - 推奨対応: ユーザー `draw()` には触れず、`LineRenderer.draw()` のレイヤーループ内で最初に `geometry = layer.geometry; if isinstance(geometry, LazyGeometry): geometry = geometry.realize()` と一度だけ実体化し、その `geometry` を `_upload_geometry(geometry)` とスナップショットの両方に使い回す（`_upload_geometry()` のシグネチャ変更も不要）。

3. **Medium – 色正規化エラーを黙殺してスタイルが落ちる**
   - `src/api/layers.py:17-24` と `src/engine/runtime/worker.py:70-90` で色正規化の例外を握りつぶし、`None` や黒にフォールバックしている。
   - 誤った色指定でもユーザーには沈黙で、レイヤーが無色扱いになるだけで原因に気づきづらい。正規化は `util.color.normalize_color()` に任せれば十分なので重複防御になっている。
   - 提案: 例外はそのまま伝播させるかログを出す共通ヘルパーに寄せ、異常入力を早期に気づけるようにする。
   - 推奨対応: `normalize_color` に任せて例外はそのまま上げる（`api.layers` では ValueError がユーザーに届き、ワーカーでは `WorkerTaskError` に包まれて表面化する）。どうしてもクラッシュ回避したい場合でも最低限 `logging.warning` を出してフォールバック色にするなど、黙殺は避ける。
4. **Low – レイヤー描画ループの広範な例外握りつぶしで不具合検知が困難**
   - `src/engine/render/renderer.py:150-178` で `set_line_color` / `set_line_thickness` を含むほぼ全処理が `try/except: pass` で囲われ、GL 更新失敗や不正 layer 値があっても静かに進む。
   - 影響: 実環境のみで起きる ModernGL 例外や型不整合を検知できず、色や太さが更新されないまま描画されるリスクがある。
   - 提案: 例外種類を絞る、もしくはデバッグレベルでログを出して原因を追えるようにする。
