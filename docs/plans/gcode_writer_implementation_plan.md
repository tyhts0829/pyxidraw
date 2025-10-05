# gcode.py 実装計画（旧 `old__gcode_generator.py` を参考）

目的: 現在のエクスポート基盤（`engine.export.service`）から利用できる、単純で堅牢な G-code 書き出し実装を `src/engine/export/gcode.py` に提供する。旧実装の有用な振る舞い（Z制御・F切替・Y反転・範囲検証・近接連結）を段階導入する。

- 対象: `src/engine/export/gcode.py`
- 参照: `src/engine/export/old__gcode_generator.py`（例: 生成手順と各設定）
  - ヘッダ/ボディ/フッタ生成: src/engine/export/old__gcode_generator.py:159
  - Z/Feed 切替: src/engine/export/old__gcode_generator.py:194,200
  - 近接連結: src/engine/export/old__gcode_generator.py:137
  - 範囲検証: src/engine/export/old__gcode_generator.py:121
  - Y 反転: src/engine/export/old__gcode_generator.py:89
  - A4 幅調整: src/engine/export/old__gcode_generator.py:76

---

## スコープと前提

- 入力は `coords: np.ndarray` と `offsets: np.ndarray`（`engine.export.service` が提供）。
  - `coords`: 形状を結合した頂点列。形状は 2D もしくは 3D（Z があれば無視）。
  - `offsets`: 各ラインの開始インデックス（累積、終端を含む）
- 出力はテキスト G-code。`ExportService` 側で `.part` → 最終 `.gcode` リネームを担当。
- 旧実装の I/O（pkl 読み込み）は非採用。座標系・単位はミリメートル前提（ヘッダで `G21`）。

不明点（要確認）
- キャンバス座標系: 現在の `coords` は画面左上原点か、機械座標（左下原点）か。
- 単位: `coords` は mm 換算済みか。変換が必要な場合の係数。
- 旧既定オフセット（`offsets=(91,-0.75)` 相当）の継続要否。
- ベッドレベリング（`M420 S1 Z10`）の有効化可否（プリンタ依存）。

---

## API 設計（最小）

既存スケルトンに準拠し、まずは最小パラメータで実装。

- `GCodeParams`（既存）
  - `travel_feed: float`（早送り）
  - `draw_feed: float`（描画）
  - `z_up: float`, `z_down: float`
  - `y_down: bool`（True で Y 反転）
  - `origin: tuple[float, float]`（全体のオフセット、旧 `offsets` 相当）
  - `decimals: int = 3`（丸め桁）

追加提案（要同意後導入）
- `bed_range: tuple[float, float]`（範囲検証、旧 `printer_range`）
- `connect_distance: float`（近接連結しきい値、旧 `connect_distance`）
- `x_adjust_mm: float` または `paper_width_mm: float`（旧 `adj_x` の一般化）
- `header_flags: dict[str, bool]`（`M420` の有効/無効など）

---

## G-code 出力仕様（旧実装ベース）

- Header（既定）
  - `G21` 単位 mm
  - `G90` 絶対座標
  - `G28` 原点復帰
  - `M107` ファンOFF
  - `M420 S1 Z10`（要確認）
- Body
  - 各ライン先頭点の前に: `Z = z_up`, `F = travel_feed`
  - 2 点目移動の前に: `Z = z_down`, `F = draw_feed`
  - 以後は `G1 X{X} Y{Y}` を逐次出力（`origin` 適用、丸め）
- Footer（最小）
  - `G1 Z{z_up}`（ペンアップ）
  - 必要なら完了コメントや `M117` 等（任意）

---

## 実装方針

- 純粋・決定的: 受け取った配列から文字列列を生成して `fp` に書き出すのみ。
- 配列前処理
  - 次元正規化: `coords[:, :2]` で 2D に落とす
  - Y 反転: `y = (canvas_h_mm - y)` もしくは `y = -y`（要件確定後）。初期は `y_down=False` 既定で保守的に。
  - オフセット: `origin = (ox, oy)` を加算
  - 近接連結: `connect_distance` 有効時のみ適用（Stage 2）
  - 範囲検証: `bed_range` 指定時のみ実施（Stage 2）
- 出力
  - 文字列は小数丸め（`decimals`）後に f-string で埋め込み
  - F 指令は必要箇所でのみ切替（冗長発行を避ける）
  - 進捗は現段階で終端のみ（`ExportService` の設計に従う）

---

## 実装ステップ（チェックリスト）

Stage 1（最小の完成）
- [ ] `write()` 本体: ヘッダ/ボディ/フッタの直列生成
- [ ] 2D/3D 入力整形と `origin` 適用、`decimals` 丸め
- [ ] Z/Feed の切替（ライン開始時と 2 点目前）
- [ ] 単体テスト（最小）: 単一ラインで期待コマンドが並ぶこと

Stage 2（旧機能パリティ）
- [ ] 近接連結（`connect_distance`）の導入（任意有効）
- [ ] 範囲検証（`bed_range`）で逸脱時に `ValueError`
- [ ] Y 反転の仕様確定と実装（`canvas_h_mm` または簡易反転）
- [ ] A4 幅調整の一般化（必要なら `x_adjust_mm`）

Stage 3（使い勝手/保守性）
- [ ] ヘッダ選択フラグ（`M420` など）とコメント整備
- [ ] Docstring（NumPy スタイル）と型注釈の充実
- [ ] ベンチ/フォーマット（`ruff/black/isort` 対象ファイルのみ）
- [ ] `ExportService` への `GCodeWriter` 差し込みテスト（e2e）

---

## テスト計画（編集ファイル優先）

- 単体（新規）
  - 入力: 2 点線分 × 2 ライン（`offsets=[0,n1,n1+n2]`）
  - 期待: ライン開始時に `Z{z_up}`+`F{travel}`、2 点目前に `Z{z_down}`+`F{draw}`、各点 `G1 X Y`
  - 丸め/オフセット/Y 反転の検証ケース
- 統合（既存サービス）
  - `ExportService(writer=GCodeWriter())` で `.part → .gcode` 完了を確認（`tmp_path`）

---

## 旧→新の対応表（主要項目）

- `GcodeConfig.offsets` → `GCodeParams.origin`
- `z_up/z_down` → 同名
- `draw_speed/travel_speed` → `draw_feed/travel_feed`
- `connect_distance` → 追加提案 `connect_distance`
- `printer_range` → 追加提案 `bed_range`
- `a4_width_mm` → 追加提案 `x_adjust_mm` または `paper_width_mm`
- `_drop_z` → `coords[:, :2]`
- `optimize_vertices_list` → 近接連結（Stage 2）
- `_generate_*` → `write()` 内部に私有関数化
- `_save_gcode` → `ExportService` が担当

---

## 確認事項（Ask-first）

- 既定の `origin` を旧 `(91, -0.75)` に合わせるか（現行プリンタで妥当か）。
- `M420 S1 Z10` をヘッダに残すかオプション化するか。
- キャンバスの高さ（Y 反転に必要）が取得可能か。無ければ反転要件自体を再確認。
- 近接連結・範囲検証は既定で無効（オプトイン）で良いか。

---

## 完了条件（この変更単位）

- `src/engine/export/gcode.py` 実装後、当該ファイルに対し `ruff/mypy/pytest`（該当テストのみ）が成功。
- 追加テストがグリーン。`ExportService` 経由の動作を手動またはスモークで確認。

以上の方針で着手可。上記「確認事項」への回答に応じて Stage 2 以降の具体値を確定します。
