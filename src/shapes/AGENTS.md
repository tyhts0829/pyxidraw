この AGENTS.md は `src/shapes/` 配下に適用されます（rev2）。

目的/役割

- プリミティブ形状（関数）とレジストリ（`shapes.registry`）。
- 生成に専念し、変換/加工/描画は他層へ委譲（分離）。

依存境界（契約）

- 依存可: `engine.core.Geometry`, `common/*`。
- 依存不可: `effects/*`, `engine/render/*`, `engine/runtime/*`, `engine/ui/*`, `engine/io/*`。
- レジストリ参照は `api/*`, `effects/*`, `shapes/*` に限定。

設計方針 / Do

- 形状関数は純粋関数（副作用なし）で `Geometry` を返す（推奨）。
- 変換（translate/scale/rotate/concat）は `engine.core.geometry.Geometry` 側で適用。shape 関数では行わない。
- 登録は `@shape`（`BaseRegistry` 準拠）を使用。対象は「関数」のみ。
- アンカー/座標規約（既定）
  - 原点中心（中心 (0,0,0)）。
  - XY 平面（Z=0）。3D 形状のみ必要に応じて Z を使用。
  - 正規化スケールで生成（例: 直径 1、振幅 1、辺長 1 相当）。実寸[mm]は後段の `Geometry.scale(...)` で付与。

パラメータ / **param_meta**（RangeHint とキャッシュ量子化）

- 公開パラメータは実値（float/int/bool/vector/str）で受け取る。
- RangeHint として `__param_meta__` を必ず書くこと。無い場合は GUI 既定レンジ（0–1）で扱う（クランプは GUI 表示上のみ）。
- `__param_meta__` の基本フィールド
  - `type`: `"number" | "integer" | "boolean" | "vector"`（任意。数値系は既定値/実引数からの型推定により省略可）
  - `min` / `max` / `step`: レンジと刻み（vector は成分ごと、`step` の成分不足は末尾値で補完）。
- 署名生成（キャッシュ鍵）では「float(Vec 要素も対象) のみ量子化」
  - 量子化刻みは `__param_meta__["step"]` を優先、未指定時は `1e-6`（環境変数 `PXD_PIPELINE_QUANT_STEP` で上書き可）。
  - int/bool は量子化せずそのまま。vector は成分ごとに量子化。
  - Shapes は鍵のみ量子化（実行引数は実値）。Effects は量子化後の値が実行引数にも渡る（設計差）。

補足（vector の扱い）

- GUI はベクトル次元を既定値（デフォルト引数）の長さから推定（最大 4）。`type:"vector"` の明示は任意。
- `min/max/step` はタプルで指定し、`step` の成分数不足は末尾値で補完される。

丸め/クランプの方針

- 実装側は「安全性のための最小限」の丸め/クランプのみ行う（例: 分割数の整数化と上下限）。
- GUI の数値入力は RangeHint による表示上のクランプを基本とする。
- 範囲外/edge 条件は「no-op 条件」や「上限クリップ」として docstring に事実として記述する。

ドキュメンテーション（docstring 規約）

- 関数 docstring（IDE 補完でのユーザー向け最小情報）
  - 構成: 先頭 1 行の要約 + Parameters のみ。
  - Parameters には「型、既定値、単位（必要時）、許容範囲/刻み、no-op 条件」を事実として記述。
  - 含めない: Returns/Notes/Examples/実装詳細/性能メモ/内部スイッチ。
  - 形式: NumPy スタイル、日本語の事実記述（主語省略・終止形）。
- モジュール docstring（shape モジュール）
  - 原則「不要」。各 shape は関数 docstring のみとする。
  - 例外: 1 モジュールに複数 shape を束ねる、または不可欠な制約を共有する場合に限り、2–3 行の最小ヘッダを許容。
- 単一情報源/同期
  - `__param_meta__` と docstring の範囲・既定値・単位を一致させる。
  - パラメータ変更時は docstring と `__param_meta__` を同時更新。公開 API 影響時はスタブ再生成/テスト更新。

命名/型（コーディング規約補足）

- Python 3.10 型ヒント必須。戻り値型は `-> Geometry` を明記。
- ファイル/モジュールは `lower_snake_case.py`、関数/変数は `snake_case`。
- 分割数系の命名は用途で一貫（例: 曲線近似は `segments`、格子は `subdivisions`）。

モジュール内の記載順

- `import` 群（標準庫 → サードパーティ → ローカルの順、空行で区切る）
- 公開の形状関数本体（`@shape` 付き、純粋、`-> Geometry`）
- `__param_meta__` の定義（公開パラメータの RangeHint/step を明記）
- ヘルパ関数・内部実装（`_` プレフィックス推奨）
  - 既存実装の一括移動は不要。新規/変更時に順次適用。

登録/公開

- `@shape` / `@shape()` / `@shape("name")` を許可。関数のみ登録可。
- `Geometry` を返すことを推奨（`Geometry.from_lines([...])` で正規化）。

テスト指針

- 代表パラメータの変化に対する性質検証
  - 頂点数、オフセット数（ポリライン本数）。
  - バウンディング（原点中心・正規化スケール内に収まること）。
  - no-op 条件（例: 分割数 < 1 → 空形状、など）。
  - 整数パラメータの丸め/上限の挙動。
- 純粋性/再現性
  - 乱数は使わない。必要なら seed を受け取り固定化。

複数 shape を同一モジュールに置く場合

- 推奨は「1 shape = 1 モジュール」。やむを得ず複数置く場合は shape 毎に
  - 形状関数本体 → 直後にその `__param_meta__` → 関連ヘルパ（`_` 付き）
    の順でまと め、共有ヘルパはファイル末尾に置く。

Don’t

- 描画/加工処理を混在させない（変換も含めて Geometry 側で行う）。
- 重い依存を追加しない。`print()` は使用しない（ログは `logging`）。

テンプレ（最小例）

先頭 1 行: 「直径 1 の円を生成。」

## Parameters

radius : float, default 0.5
正規化半径。許容 [0, 0.5]。0 で空形状。
segments : int, default 64
円弧近似の分割数。許容 [3, 1024]。範囲外は安全側に丸め/上限。

`__param_meta__` 例

```python
circle.__param_meta__ = {
    "radius": {"type": "number", "min": 0.0, "max": 0.5, "step": 1e-4},
    "segments": {"type": "integer", "min": 3, "max": 1024, "step": 1},
}
```

備考

- 既定では正規化スケールで生成し、必要に応じて `Geometry.scale(mm)` で物理単位を与える。
- GUI は「draw 内で未指定（既定値採用）の引数のみ」を表示・調整し、優先順位は「明示引数 > GUI > 既定値」。
