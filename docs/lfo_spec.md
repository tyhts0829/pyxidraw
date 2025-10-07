# LFO（低周波オシレータ） 仕様案 / 実装計画

本書は、`from api import lfo` で利用できる時間依存の LFO（Low Frequency Oscillator）オブジェクトの最小で明快な仕様を定義し、段階的な実装計画（チェックリスト）を示す。
目的は「時間 `t`（秒）を入力として 0.0〜1.0 の範囲で振動する値を返す」単純・決定的・再利用可能な部品を提供すること。

---

## 目的 / スコープ
- 目的: スケッチ内の任意パラメータを時間的に変調するための純粋関数的な LFO を提供。
- スコープ: 波形生成（サイン/トライアングル/ノコギリ/スクエア/パルス/S&H/Perlin）。範囲スケーリング、位相、周期/周波数。
- 非スコープ: 複雑なエンベロープ、外部同期（MIDI/OSC）、波形合成、波形のモーフ、GUI 連携（将来拡張の余地は残す）。

## ユーザー体験（使用例）
```python
from api import lfo

# 構成済み LFO を取得して、任意の t で評価
osc = lfo(wave="sine", freq=0.5, phase=0.0)  # 0.5 Hz（2秒周期）、0..1 を往復
v = osc(t)  # => 0.0 .. 1.0 の float

# 出力レンジを変更（例: 0.2..0.8 に制限）
osc2 = lfo(wave="triangle", freq=1.0, lo=0.2, hi=0.8)

# パルス幅 30% の矩形波
gate = lfo(wave="pulse", freq=2.0, pw=0.3)

# サンプル&ホールド（ランダム）— 周期ごとに一定値を維持（決定的シード）
sh = lfo(wave="sh", freq=1.0, seed=123)
 
# Perlin ノイズ LFO（滑らかな連続ノイズ）。周波数は時間に対する変化の速さ。
noise = lfo(wave="perlin", freq=0.2, octaves=3, persistence=0.5, lacunarity=2.0, seed=42)
```

---

## 公開 API（最小）
- エクスポート: `api.lfo`
- シグネチャ（案）:

```python
def lfo(
    wave: str = "sine",       # "sine" | "triangle" | "saw_up" | "saw_down" | "square" | "pulse" | "sh" | "perlin"
    *,
    freq: float | None = 1.0,  # 周波数 [Hz]。`period` と同時指定不可（Perlin は周期性を持たないが速度尺度として利用）
    period: float | None = None,# 周期 [秒]。指定時は freq を無視（Perlin では内部で freq=1/period を速度として利用）
    phase: float = 0.0,         # 位相 [周期単位]。0..1 が 1 周（1 以上可、内部で wrap）。Perlin は時間オフセットとして加算。
    lo: float = 0.0,            # 出力最小（既定 0.0）
    hi: float = 1.0,            # 出力最大（既定 1.0）
    pw: float = 0.5,            # パルス幅（矩形/パルスのみ 0..1）
    skew: float = 0.0,          # 歪み（-1..1）。三角/ノコギリ系で傾きを変化
    seed: int | None = None,    # SH/Perlin 用の決定的シード（None なら固定シード）
    # Perlin ノイズ専用（fBm）
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> "LFO":
    """LFO を構成して返す。返り値は `__call__(t: float) -> float` を持つ。

    何を: 時間 t [sec] に対して 0..1 の値を返す LFO を生成。
    単位: `freq` は Hz、`period` は秒、`phase` は周期単位（1.0 で 1 周）。
    決定性: 同一引数と t に対して常に同じ値を返す（S&H/Perlin は seed に依存）。
    注意: Perlin は厳密な周期性を持たない。`freq`/`period` は時間―ノイズ空間のスケール（変化の速さ）。
    """
```

- 返り値: `LFO`（`Callable[[float], float]` な軽量オブジェクト）。
- 例外: 不正な引数（`hi <= lo`、`freq <= 0`/`period <= 0`、`pw` が [0,1] 外など）は `ValueError`。
- 安全: 出力は最終的に `clamp(lo, hi)` を保証。

補足（設計意図）
- `api` は薄いファサードを維持するため、波形ロジックは `src/common/lfo.py` に実装し、`api.lfo` はファクトリ再エクスポートのみとする。
- `phase` は「周期単位（回転数）」で統一し、`phi = frac(freq*t + phase)` を基本形とする。

---

## 波形仕様（正規化位相 phi = frac(freq*t + phase) を前提）
- 正規化: 各波形は一旦 [-1, 1] の双極値を生成後、`y = (y + 1) * 0.5` で 0..1 化し、最後に `lo..hi` へ射影。

- sine: `y = sin(2π * phi)`（滑らか、対称）
- triangle: 対称三角（`skew=0`）。`skew` で上り/下りの比率を連続変形（例: `skew>0` で上りが短く下りが長い）。
- saw_up: `y = 2*phi - 1`（上りノコギリ）。`skew` は `phi` の非線形写像で傾斜を変化。
- saw_down: `y = 1 - 2*phi`（下りノコギリ）。`skew` の適用は `saw_up` と同様。
- square: `y = 1` if `phi < 0.5` else `-1`。`pw` は 0.5 固定（矩形）。
- pulse: `y = 1` if `phi < pw` else `-1`（`pw`∈(0,1)）。
- sh（Sample & Hold）: 区間ごとに一定値。`k = floor((t/period) + phase)` をインデックスに `hash(seed,k)` 由来の `u∈[0,1]` を生成し、`y = 2*u - 1`。

- perlin: 1D Perlin/勾配ノイズの fBm（fractal Brownian motion）。連続的で滑らか、非周期。
  - 時間→ノイズ座標: `x = freq*t + phase`（`period` 指定時は `freq=1/period`）。
  - 基本ノイズ: `perlin1(x, seed)` は [-1,1]。フェード関数 `fade(u)=6u^5-15u^4+10u^3` を使用。
  - fBm 合成: `sum_{i=0..oct-1} perlin1(x * lacunarity^i) * (persistence^i)` を [-1,1] に正規化後、0..1 化。
  - シード: 安定な固定置換テーブル（seed によりスクランブル）。Python の `hash()` 非依存で決定的。

`skew` の正規化（例）
- `skew ∈ [-1,1]` を `s = (skew+1)/2` に写像し、`phi' = phi**gamma` または逆冪で連続変形。
- 近似: `gamma = lerp(0.25, 4.0, s)`。`skew=0` で `gamma=1`（等速）。

---

## 実装配置 / 依存
- 配置
  - `src/common/lfo.py`: 波形関数と `LFO` クラス本体（純粋ロジック）。
  - `src/api/lfo.py`: ファクトリ関数 `lfo(...) -> LFO`（薄い再エクスポート）。
  - `src/api/__init__.py`: `from .lfo import lfo` を追加して公開。
- 依存
  - 標準ライブラリ（`math`, `typing`）。外部依存なし。
  - `api`→`common` の依存は architecture.md の方針に適合。

---

## 仕様詳細（入力検証 / 数値仕様）
- 時間 `t`: float（秒）。負の `t` も許容（位相 wrap で自然に対処）。
- 周波数/周期: `freq` と `period` はどちらか一方。`period` 優先で `freq = 1/period`。
- 位相: 任意実数。内部で `frac()` により wrap。
- 範囲: `lo < hi` を要求。出力は最後に `min(max(y, lo), hi)` で保証。
- S&H/Perlin の決定性: `seed` が None の場合も固定既定値で決定的。`hash/置換テーブル` は自前実装で安定化（Python の `hash()` に依存しない）。
- 演算誤差: 0..1 への丸めで ULP 漏れを抑制（`y = min(max(y, 0.0), 1.0)` の最終ガード）。

---

## ドキュメンテーション / メタ
- docstring: NumPy スタイル（日本語、事実記述）。
- 型ヒント: Python 3.10 範囲。`typing` は最小限。
- `__param_meta__`（補助、RangeHint）: GUI 直結ではないが将来の統一のため付与（例）

```python
__param_meta__ = {
    "wave": {"type": "enum", "choices": ["sine","triangle","saw_up","saw_down","square","pulse","sh","perlin"]},
    "freq": {"type": "float", "min": 1e-4, "max": 10.0, "step": 1e-3},
    "period": {"type": "float", "min": 0.01, "max": 60.0, "step": 1e-3},
    "phase": {"type": "float", "min": 0.0, "max": 1.0, "step": 1e-3},
    "lo": {"type": "float", "min": 0.0, "max": 1.0, "step": 1e-3},
    "hi": {"type": "float", "min": 0.0, "max": 1.0, "step": 1e-3},
    "pw": {"type": "float", "min": 0.01, "max": 0.99, "step": 1e-3},
    "skew": {"type": "float", "min": -1.0, "max": 1.0, "step": 1e-3},
    # Perlin ノイズ用
    "octaves": {"type": "int", "min": 1, "max": 8},
    "persistence": {"type": "float", "min": 0.0, "max": 1.0, "step": 1e-3},
    "lacunarity": {"type": "float", "min": 1.0, "max": 8.0, "step": 1e-3},
}
```

量子化（署名生成）
- `params_signature` を用いる箇所では float のみ step に従い量子化（既定 1e-6）。LFO はランタイム評価用の純関数であるため、キャッシュ鍵に組み込む場合のみ適用。

---

## テスト計画（最小）
- 位置: `tests/api/test_lfo.py`
- 項目
  - 値域: 全波形で `t` を代表値サンプル（100 点）して `lo <= y <= hi` を満たす。
  - 周期性: 周期波形（sine/triangle/saw/square/pulse/sh）で `y(t) ≈ y(t+period)` を確認。Perlin は対象外。
  - パラメータ効果: `pw`/`skew` が期待方向に単調変化すること。
  - 決定性: 同一 `seed` で S&H/Perlin の系列一致、`seed` 違いで非一致。
  - Perlin 連続性: 微小 Δt に対し |y(t+Δt)-y(t)| が小さい（freq に応じて上限を設定）。
  - エラー: 不正引数で `ValueError`。

---

## 実装計画（チェックリスト + 合格基準）

前提/制約:
- 依存追加なし（Ask-first 遵守）。
- 変更ファイルに限定して `ruff/mypy/pytest` を回す（AGENTS.md 方針）。
- 公開 API 追加のためスタブ再生成を行う。

Stage 0 — スケルトン
- [ ] `src/common/lfo.py` を追加（空の実装＋型/ヘッダ）。
- [ ] `src/api/lfo.py` を追加（`from common.lfo import LFO, lfo`）。
- [ ] `src/api/__init__.py` に `lfo` を再エクスポート。
- 合格基準: import/型が通る（実体は `NotImplementedError` でも可）。

Stage 1 — 波形コア
- [ ] 正規化位相 `phi` の計算、`sine/triangle/saw_up/saw_down/square/pulse` を実装。
- [ ] `skew` の非線形写像（`gamma` 方式）を実装し、適用対象波形へ反映。
- [ ] `lo..hi` 射影と最終 clamp、入力検証（`ValueError`）。
- 合格基準: 簡易サンプルで 0..1 を維持、周期性の目視検証。

Stage 2 — S&H / Perlin（決定的）
- [ ] S&H: 軽量ハッシュ（LCG など）により `k`→`u∈[0,1]` を決定的生成。
- [ ] Perlin: 1D 勾配ノイズ `perlin1(x, seed)` を実装（固定置換テーブル + fade + 勾配補間）。
- [ ] Perlin fBm: `octaves/persistence/lacunarity` を適用し [-1,1] 正規化後に 0..1 射影。
- [ ] `seed=None` でも固定既定値で決定的に。
- 合格基準: 同一パラメータで再現、`seed` 違いで非一致。Perlin は連続で、過度なスパイクがない。

Stage 3 — 単体テスト
- [ ] `tests/api/test_lfo.py` を追加（上記テスト計画）。
- 合格基準: 変更ファイル限定の `pytest -q tests/api/test_lfo.py` 緑。

Stage 4 — ドキュメント/スタブ
- [ ] `api` の docstring 整備（NumPy スタイル、日本語）。
- [ ] `python -m tools.gen_g_stubs` で `src/api/__init__.pyi` を更新し差分ゼロ。
- 合格基準: `tests/stubs/test_g_stub_sync.py` 緑（必要なら）。

Stage 5 — 整合/仕上げ
- [ ] `architecture.md` に API 追加の要点（参照先: `src/common/lfo.py`）を追記。
- [ ] 変更ファイルに対して `ruff/black/isort/mypy` 実行。
- 合格基準: Lint/Format/Type OK。簡易スモーク `pytest -q -m smoke` または対象テスト緑。

---

## 非目標 / 将来拡張メモ
- BPM/拍同期（`tempo`/`beats`）の砂糖 API。
- エッジのスムージング（スクエア/パルス用のスルー/双曲線タンジェント混合）。
- OpenSimplex/Periodic Perlin の追加、2D/3D ノイズによる時間走査の拡張。
- GUI からの可視化/試聴ウィジェット（DPG の補助）。

---

## 受け入れ基準（DoD）
- `from api import lfo` でファクトリが利用可能。
- `osc = lfo(...); osc(t)` が 0..1 を返し、主要波形が意図通り動作。
- 公開 API スタブが最新、`ruff/black/isort/mypy` と対象テストが緑。
