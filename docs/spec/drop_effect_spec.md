# drop エフェクト 仕様案 / アイデアメモ

本書は、線や面の一部を「間引く」ためのエフェクト `drop` の仕様アイデアを整理するメモである。
目的は「ジオメトリを単純にしつつ、ランダム性やリズム感のある抜け」を作るシンプルな加工を提供すること。

---

## ゴール / コンセプト

- 線（ポリライン）や「面相当」（閉じた線）を、規則的または条件付きで削除する。
- 幾何を複雑に変形しない（頂点座標は変更しない）。やることは「線/面を丸ごと残すか捨てるか」に限定する。
- 単純なルール（間引き・長さ・ランダム）を組み合わせて、スケッチごとのリズムを作りやすくする。

---

## ユースケース例

- 密なハッチングやグリッドから、一部の線だけを抜いて「スカスカ感」「ノイズ感」を足す。
- 面をランダムに落として、メッシュ構造やドット絵のような抜けを作る。
- 「一定ステップごと」「一定長未満/超の線だけ」など、明快なルールで構造を削ぎ落とす。

---

## 公開 API イメージ

```python
from api import E

pipe = (
    E.pipeline
    .drop(
        interval=2,         # 2 本ごとに 1 本 drop
        offset=0,           # どこから数え始めるか
        min_length=None,    # この長さ未満は drop
        max_length=None,    # この長さより長いものを drop
        probability=0.0,    # 各線を確率的に drop
        by="line",          # "line" | "face"
        seed=None,          # ランダム drop の決定性
        keep_mode="keep",   # "keep" | "drop"（条件に合致したものを残すか捨てるか）
    )
    .build()
)

g2 = pipe(g1)
```

※実装時は `by="line"` のみを最初に対応し、「閉じた線（先頭と末尾が同一点）」を `face` 相当として扱うなど、必要に応じて段階的に拡張する想定。

---

## コア機能アイデア

### 1. インデックス間引き（interval / offset）

- `interval: int | None`  
  - `interval=2` なら「線インデックスを 0,1,2,3,... としたときに、2 本ごとに 1 本を drop」する。
  - 具体例（`offset=0` のとき）:
    - 線 0: keep
    - 線 1: drop
    - 線 2: keep
    - 線 3: drop
    - ...
  - 実際のルール例:
    - `"keep"` モード: `(i - offset) % interval == 0` の線だけ残す。
    - `"drop"` モード: `(i - offset) % interval == 0` の線だけ捨てる。
- `offset: int`  
  - どの線インデックスからカウントを始めるかをずらす。
  - 例: `interval=3, offset=1, keep_mode="drop"`  
    - `(i - 1) % 3 == 0` の線（1,4,7,...) を drop。

### 2. 長さ条件（min_length / max_length）

- `min_length: float | None`  
  - 指定した場合、線長 `L` が `L <= min_length` の線を drop する（または `"keep"` モードなら逆に「その線だけ残す」）。
  - 例: `min_length=2.0, keep_mode="drop"`  
    - 2 以下の短い線をすべて削除し、細かいノイズ線を掃除する。
- `max_length: float | None`  
  - 指定した場合、`L >= max_length` の長い線を drop する。
  - 長い輪郭線だけ落として、細かいディテール線のみ残すといった使い方を想定。
- 長さの定義:
  - 各ポリラインについて、連続する頂点間のユークリッド距離を合計したものを線長とする。

### 3. 確率的 drop（probability / seed）

- `probability: float`  
  - `0.0`〜`1.0` の範囲で、各線を drop する確率を指定。
  - 例: `probability=0.3`  
    - 各線ごとに独立に 30% の確率で削除される。
  - `interval` や `min_length` と組み合わせて、「一定パターン＋ランダム崩し」を作る。
- `seed: int | None`  
  - None の場合は内部既定 seed で決定的に動作。
  - 同じ入力 Geometry と同じパラメータなら、常に同じ線が drop される。

### 4. 単位の選択（by）

- `by: Literal["line", "face"]`
  - `line`: Geometry の offsets に基づく各ポリラインを drop 単位とする。
  - `face`: 「閉じた線（最初と最後の頂点が同じ）」「あるいは将来的な面構造」がある場合に、面単位で drop するモード。
  - 初期実装では `line` のみを正式サポートし、`face` は後方互換を壊さずに拡張可能な将来オプションとして扱う。

### 5. 条件の組み合わせと keep/drop モード

- `keep_mode: Literal["keep", "drop"]`
  - `"drop"`:
    - 条件に合致した線を drop（削除）し、その他をそのまま残す。
    - 条件は「OR 結合」を基本とし、どれか 1 つでも drop 条件を満たせば削除。
  - `"keep"`:
    - 条件に合致した線だけを残し、それ以外を drop する。
    - 「この条件にマッチする線だけ抽出したい」用途に使う。
- 条件組み合わせイメージ:
  - 条件群: `C_interval`, `C_min_length`, `C_max_length`, `C_probability`
  - 内部で `C = C_interval or C_min_length or C_max_length or C_probability` のような形でまとめる。

---

## デフォルトと安全な初期値

- 既定値は「何もしない」ことを優先する。
  - `interval=None`
  - `offset=0`
  - `min_length=None`
  - `max_length=None`
  - `probability=0.0`
  - `by="line"`
  - `seed=None`
  - `keep_mode="drop"`
- すべての条件が無効（`interval is None` かつ `min_length/max_length is None` かつ `probability == 0.0`）の場合は no-op（入力 Geometry をそのまま返す）。

---

## GUI / パラメータメタデータ案

`__param_meta__` のイメージ:

```python
__param_meta__ = {
    "interval": {"type": "int", "min": 1, "max": 100, "step": 1},
    "offset": {"type": "int", "min": 0, "max": 100, "step": 1},
    "min_length": {"type": "float", "min": 0.0, "max": 1000.0, "step": 0.1},
    "max_length": {"type": "float", "min": 0.0, "max": 1000.0, "step": 0.1},
    "probability": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "by": {"type": "enum", "choices": ["line", "face"]},
    "keep_mode": {"type": "enum", "choices": ["keep", "drop"]},
    "seed": {"type": "int", "min": 0, "max": 2**31 - 1},
}
```

GUI 上では、まずは `interval`, `min_length`, `probability` の 3 つを主役とし、その他は「詳細設定」として折りたたむイメージ。

---

## 実装に向けたメモ（簡易）

- 入力: `Geometry`（`coords`, `offsets`）を受け取り、`offsets` の区間ごとに判定する。
- 各線の長さは `coords[offsets[i]:offsets[i+1]]` の距離和として計算。
- 判定結果を bool マスク配列として持ち、残す線の `coords` をまとめて再構成して新しい `Geometry` を返す。
- ランダム判定は `seed` から `np.random.Generator` を生成し、線インデックスに対して 1 回だけ使用（決定的）。

---

## 要確認事項（今後相談したい点）

- `interval` と `probability` を同時に指定したときの優先度（単純に OR 結合で良いか）。
- 長さの単位（mm 相当とするか、単純な座標系距離とみなすか）の明示。
- `face` モードを Geometry のどの仕様に紐づけるか（閉じた線の扱い、将来のポリゴン表現との整合）。
