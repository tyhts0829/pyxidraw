# PyXidraw6

このリポジトリは、ラインベースの幾何生成とエフェクト処理を行い、リアルタイムに描画するための軽量フレームワークです。公開 API を通じて、形状 `G.<name>()` とパイプライン `E.pipeline.<effect>(...).build()` を組み合わせたスケッチを簡潔に記述できます。

重要（v6.0 以降の破壊的変更）:
- Shape は関数ベースに統一されました。`@shape def polygon(...)->Geometry` のように登録してください。
- 旧来の `BaseShape` 継承は撤廃されました。
- ユーザー拡張の登録経路は `from api import shape`（唯一）です。

- 目的/全体像: `docs/architecture.md`
- パイプライン/API: `docs/pipeline.md`
- エフェクト一覧: `docs/effects.md`
- シェイプ一覧: `docs/shapes.md`
- LFO 仕様: `docs/lfo_spec.md`
- 開発環境セットアップ: `docs/dev-setup.md`
- コーディング規約/運用ガイド: `AGENTS.md`

## クイックスタート

1) Python 3.10+ を用意し、仮想環境を作成:
```
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
```

注記: ModernGL は必須依存です（`pip install -e .[dev]` または `pip install -e .` で自動的に導入されます）。ヘッドレス検証のみを行う場合は `run_sketch(..., init_only=True)` を使用してください（GL を初期化しません）。

2) スタブ生成とスモークテスト:
```
PYTHONPATH=src python -m tools.gen_g_stubs
pytest -q -m smoke
```

3) 実行例（プレビュー）:
```
python main.py
```

HUD 設定の例（任意）:
```
from engine.ui.hud import HUDConfig
from api.sketch import run_sketch

# CACHE 表示を有効化し、CPU/MEM を無効化
cfg = HUDConfig(show_cache_status=True, show_cpu_mem=False)

run_sketch(user_draw, hud_config=cfg)
```

Parameter GUI からの HUD 切替（任意）:

- `use_parameter_gui=True` かつ `show_hud=None` のとき、GUI に `Show HUD`（`runner.show_hud`）が現れます。チェックで HUD の表示/非表示を動的に切り替えられます。
- `show_hud` を明示（True/False）した場合は引数が優先され、GUI のトグルは無効（ロック）となります。

独自 Shape の登録例（最小）:
```
from api import shape, G
from engine.core.geometry import Geometry
import numpy as np

@shape
def ring(*, r: float = 60.0, n: int = 200) -> Geometry:
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    xy = np.c_[r*np.cos(th), r*np.sin(th)]
    return Geometry.from_lines([xy])

g = G.ring(r=80)
```

### 時間変調（LFO）の使用例
```
from api import lfo, cc

osc = lfo(wave="sine", freq=0.25)  # 4秒周期で 0..1 を往復

def user_draw(t):
    r = 40 + 20 * osc(t) + 10 * cc[1]  # LFO と CC を併用
    # ... r を使って Geometry を生成 ...
```

## 任意依存（実機/高速化など）

パラメータ GUI / cc（現行仕様の要点）
- cc は `api.cc` のグローバル辞書（`cc[i] -> float(0..1)`、未定義は 0.0）。`draw(t)` 内で自由に数値式に利用。
- GUI は「draw 内で未指定（＝既定値採用）の引数のみ」を対象に表示・調整。
- 優先順位は「明示引数 > GUI > 既定値」。MIDI→GUI の自動上書きは行わない。
- RangeHint は `__param_meta__` がある場合のみ使用し、無い場合は 0–1 既定レンジ（クランプは表示上のみ）。

- 追加インストール: `pip install -e .[optional]`
- 使用例: shapely, numba, mido, fonttools などを使う optional テストが有効になります。

## 設定ファイル

- 既定値: `configs/default.yaml`
- ローカル上書き: ルート `config.yaml`（存在すれば既定値にマージ）
- ローダ: `util.utils.load_config()`
- サンプル: `configs/example.yaml`

## リンク集

- スタブ検証 CI: `.github/workflows/verify-stubs.yml`

バグ報告/提案は Issue にお願いします。スタブ更新が必要な変更では `python -m tools.gen_g_stubs` を実行し、`api/__init__.pyi` を更新してください。
