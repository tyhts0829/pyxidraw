% エフェクト `style` 導入計画（color/thickness + 後勝ち + 即時反映）

目的
- 線色（RGB 0..1）と線太さ倍率（1.0–10.0）を 1 つの Effect `style` として提供。
- 1 パイプライン内に複数の `style` がある場合は「後勝ち」。
- Parameter GUI からの変更を描画に即時反映（色: カラーピッカー 0–255、太さ: 小数スライダー）。
- `draw` の戻り値拡張: `Geometry | LazyGeometry | Sequence[Geometry|LazyGeometry]`。

非目標
- 既存の Geometry/Shader 形式を変えない（薄いレイヤー描画で実現）。
- 既存 Effect/Shape 実装の挙動を変えない（互換）。

設計方針（概要）
- `style` は「非幾何（描画指示）」とし、幾何には手を触れない。
- ワーカ側で `draw()` の戻りを正規化し、各要素から `style` を抽出 → `StyledLayer` に変換してメインへ送る。
- Renderer は `StyledLayer[]` を受けた場合、レイヤー単位で `set_line_color`/`set_line_thickness`→`upload`→`draw` を順次実行。
- 「後勝ち」は LazyGeometry.plan の `style` ステップ列から最後のものを採用する実装で担保。
- 幾何計算のキャッシュ効率を維持するため、`style` ステップは実体化前に plan から除去して Geometry を生成（スタイル変更だけで再計算を誘発しない）。

仕様要約（要件反映）
- Effect 名: `style(color: vec3, thickness: float=1.0)`
  - `color`: 0..1 の RGB（アルファ不要）。UI は 0..255 カラーピッカー。
  - `thickness`: 1.0–10.0（現行の線太さに対する倍率）。
- 複数 `style` の解決: 後方の指定が前方を上書き。
- Parameter GUI: `color` は runner.line_color と同 UI、`thickness` は小数スライダー。変更は即時反映。
- draw 返り値: `Geometry | LazyGeometry | Sequence[Geometry|LazyGeometry]`。シーケンスは「複数スタイルの独立描画」。

実装ステップ（チェックリスト）

1) Effect 本体（登録・メタ）
- [x] 追加: `src/effects/style.py`
  - [x] `@effect()` で登録された関数 `def style(g: Geometry, *, color: tuple[float,float,float] | None = None, thickness: float = 1.0) -> Geometry:`（戻り値は g をそのまま返す no-op）。
  - [x] `__param_meta__` を付与:
        - `color`: `{type: "vec3", min: (0,0,0), max: (1,1,1), step: (1/255,1/255,1/255)}`
        - `thickness`: `{type: "number", min: 1.0, max: 10.0, step: 0.01}`
  - [x] 特徴付け: `style.__effect_kind__ = "style"` などのマーカー属性を設定（将来の検出容易化）。
  - [x] ドキュメント（docstring）: 日本語で簡潔に仕様・後勝ち・UI 想定を記載。

2) Parameter GUI（UI/監視の拡張）
- [x] DPG: `src/engine/ui/parameters/dpg_window.py` に `effect@{pipeline}.style#N.color` を特別扱いし、`runner.line_color` と同じ `dpg.add_color_edit` を使用（0–255 UI, α非表示）。
  - [x] 既存の `store_rgb01(pid, app_data)` を流用し、`pid` に effect の descriptor ID をそのまま渡して 0..1 RGBA（α=1 固定）で保存。
  - [x] `thickness` は既存スライダー生成経路（float）を利用。RangeHint 1.0–10.0。
- [ ] 監視: `src/api/sketch_runner/params.py` に `subscribe_style_changes(...)` を追加。
  - [ ] ParameterStore の変更 ID をフィルタ（`effect@.*\.style#\d+\.(color|thickness)`）。
  - [ ] front が「単一 Geometry」の場合のみ、Renderer のグローバル `set_line_color`/`set_line_thickness` を即時更新（レイヤー受信時は各レイヤー描画時に反映されるため不要）。

3) draw 返り値の拡張（ランタイム）
- [x] 型更新: `src/engine/runtime/worker.py` の `draw_callback` 型/処理を `Geometry | LazyGeometry | Sequence[...]` に拡張。インライン/サブプロセス双方。
  - [x] 受け取った戻り値を正規化: `items = list(result) if isinstance(result, Sequence) else [result]`。
- [x] `style` 抽出: 各 `item` が `LazyGeometry` の場合、`plan` を走査して `impl.__name__ == "style"`（および `getattr(impl, "__effect_kind__", None) == "style"` を優先）に該当する最後のステップを抽出。
  - [x] 幾何用 plan = `plan` から `style` ステップを除去 → `LazyGeometry(base, base_payload, filtered_plan)`。
  - [x] `StyledLayer` を構築: `geometry=LazyGeometry/Geometry`, `color=(r,g,b,1.0) | None`, `thickness=float(th) | None`。
  - [x] `item` が `Geometry` の場合は `StyledLayer` をデフォルトスタイルで包む。
- [x] `RenderPacket` を拡張: `geometry: Geometry | None`, `layers: tuple[StyledLayer, ...] | None`。後方互換: 既存の引数順を維持。

4) バッファ/受信（SwapBuffer/StreamReceiver）
- [x] `src/engine/runtime/buffer.py`: `push()`/`get_front()` の型を `Geometry | LazyGeometry | Sequence[StyledLayer] | None` に拡張。
- [x] `src/engine/runtime/receiver.py`: `RenderPacket.layers` があれば `SwapBuffer.push(layers)`、無ければ `geometry` を従来どおり push。

5) Renderer（レイヤー描画対応 + 太さ即時反映）
- [x] 追加: `src/engine/render/types.py` に `@dataclass class StyledLayer: geometry: Geometry|LazyGeometry; color: RGBA|None; thickness: float|None`。
- [x] `src/engine/render/renderer.py` を拡張:
  - [x] `set_line_thickness(self, value: float)` を追加（uniform `line_thickness` を更新）。
  - [x] `tick()`: front が `Sequence[StyledLayer]` の場合は内部バッファに保持（`self._frame_layers`）し、単一 Geometry のときは従来通り。
  - [x] `draw()`: `self._frame_layers` があれば各レイヤーで `set_line_color`→`set_line_thickness(base*layer.thickness)`→`_upload_geometry(layer.geometry)`→`vao.render()` を順に実行。完了後にクリア。
  - [x] HUD 計数はレイヤー合算で更新。

6) 後勝ち（仕様の担保）
- [x] `style` 抽出時に plan の末尾から走査して最初に見つかったものを採用（最後の指定が勝つ）。
- [x] 同一パイプライン内に複数 `style` があっても幾何用 plan からは全て除去。

7) キャッシュ/性能
- [x] `style` は Geometry 計算に影響しないため、抽出後に plan から除去して実体化 → スタイル変更のみでは Prefix/Shape キャッシュを無効化しない。
- [ ] 署名（`lazy_signature_for`）は `style` 除外後の plan を基準に計算されるようにワーカ側で再構築（必要最小）。

8) スタブ/型/API ドキュメント
- [ ] `src/api/__init__.pyi` の `run_sketch` doc と `user_draw` 仕様を更新（戻り値拡張）。
- [ ] `tools/gen_g_stubs.py` に変更が必要な場合は追従。生成コマンド: `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`。
- [ ] `architecture.md` に「レイヤー描画（StyledLayer）」「style エフェクト（非幾何・後勝ち）」を追記。

9) テスト（変更ファイル限定で高速）
- [ ] 単体: `tools/list_param_meta.py` で `style` のメタが列挙されること（color vec3, thickness range）。
- [ ] 単体: `ParameterValueResolver` が `effect@*.style#N.(color|thickness)` を Descriptor 登録すること。
- [ ] 単体: `worker` の正規化ロジックが `Sequence[LazyGeometry]` → `layers` に変換し、`style` を後勝ち抽出・幾何 plan から除去すること（GL 依存なしで検証）。
- [ ] 可能なら統合: `LineRenderer` にモック/ダミー ctx を与え、`layers` で `set_line_color`/`set_line_thickness` が順次呼ばれること（`-m integration`）。

10) 互換性/移行
- [ ] 既存スケッチ（単一 Geometry 返却）はそのまま動作。
- [ ] `runner.line_color`/既定太さは「レイヤー無し」または `style` 未指定時のデフォルトとして適用。

リスク/注意
- Renderer でレイヤーごとに `upload→render` を行うため描画回数が増える（レイヤー数に比例）。層数は数本を想定。
- Parameter GUI の effect 項目が増えると UI が煩雑化する可能性。カテゴリ分け（`effect@{pipeline}`）で整理。

完了条件（DoD）
- [ ] `style` エフェクトが登録され、Parameter GUI に `color`（カラーピッカー）/`thickness`（1.0–10.0）が表示される。
- [ ] `draw()` が `Sequence[...]` を返した場合に複数レイヤーが順描画され、色/太さが適用される。
- [ ] `style` の変更のみでは幾何の再計算が走らない（Prefix/Shape キャッシュが維持）。
- [ ] 既存の単一 Geometry 返却スケッチが従来どおり描画される。

影響ファイル（予定）
- src/effects/style.py
- src/engine/ui/parameters/dpg_window.py
- src/api/sketch_runner/params.py
- src/engine/runtime/worker.py
- src/engine/runtime/packet.py
- src/engine/runtime/buffer.py
- src/engine/runtime/receiver.py
- src/engine/render/types.py
- src/engine/render/renderer.py
- src/api/__init__.pyi
- tools/gen_g_stubs.py（必要時）
- architecture.md

開発・検証コマンド（編集ファイル限定）
- Lint/Format: `ruff check --fix {changed_files} && black {changed_files} && isort {changed_files}`
- Type: `mypy {changed_files}`
- Test（ポイント）:
  - `pytest -q -k style`（追加テストがある場合）
  - `pytest -q -m smoke`（軽確認）

Open Questions（要確認）
- `thickness` 倍率の解釈: `base` は `run(..., line_thickness)` で与えた値とする想定で良いか。
- Parameter GUI の `style.color` 即時反映は「レイヤー無し時のみグローバル適用」で問題ないか（レイヤー有り時は各レイヤー描画時に反映）。
- 0–255 UI のアルファは固定 255（1.0）でよいか。

備考
- 既存の `docs/plans/251101_per_shape_style.md`（boldify+colorize案）とは別系統の実装（Renderer レイヤー描画は踏襲）。`style` は太さを GPU uniform 倍率で扱い、幾何には触れない。
