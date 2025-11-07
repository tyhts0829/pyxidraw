# 環境変数の集中管理（common.settings）計画・チェックリスト

目的
- 環境変数の定義・既定・型・説明を一元化し、参照側の `os.getenv` 散在を解消する。
- 既定値/型の不一致や import タイミング依存の不安定さを排除する。

スコープ/非スコープ
- スコープ: ランタイムで使用される `PXD_*/PYX_*` 系の環境変数。
- 非スコープ: OS 依存の `WINDIR`、pytest 内のテスト専用フラグ（`PXD_UPDATE_SNAPSHOTS` など）。

設計方針
- モジュール: `src/common/settings.py`
  - 依存: 既存 `common.env.env_int/env_bool` を活用（実装を複製しない）。
  - 読み出し: import 時に一括ロード。`reload_from_env()` で再読み込み可。
  - 公開: 型付きフィールド（例: `PIPELINE_QUANT_STEP: float`）。`as_dict()` で辞書化。

API（案）
```python
# src/common/settings.py（案）
from dataclasses import dataclass
from common.env import env_int, env_bool
import os

@dataclass
class _Settings:
    PIPELINE_QUANT_STEP: float = 1e-6
    IBO_FREEZE_ENABLED: bool = True
    IBO_DEBUG: bool = False
    INDICES_CACHE_ENABLED: bool = True
    INDICES_CACHE_MAXSIZE: int | None = 64
    INDICES_DEBUG: bool = False
    SHAPE_CACHE_MAXSIZE: int | None = 128
    PREFIX_CACHE_ENABLED: bool = True
    PREFIX_CACHE_MAXSIZE: int | None = 128
    PREFIX_CACHE_MAX_VERTS: int = 10_000_000
    DEBUG_PREFIX_CACHE: bool = False
    COMPILED_CACHE_MAXSIZE: int | None = 128
    DEBUG_FONTS: bool = False
    USE_NUMBA: bool = True

_settings = _Settings()

def reload_from_env() -> None:
    global _settings
    # 例: float は明示処理
    try:
        q = os.getenv("PXD_PIPELINE_QUANT_STEP")
        _settings.PIPELINE_QUANT_STEP = float(q) if q is not None else 1e-6
    except Exception:
        _settings.PIPELINE_QUANT_STEP = 1e-6
    # bool/int 系はヘルパを使用
    _settings.IBO_FREEZE_ENABLED = env_bool("PXD_IBO_FREEZE_ENABLED", True)
    _settings.IBO_DEBUG = env_bool("PXD_IBO_DEBUG", False)
    _settings.INDICES_CACHE_ENABLED = env_bool("PXD_INDICES_CACHE_ENABLED", True)
    _settings.INDICES_CACHE_MAXSIZE = env_int("PXD_INDICES_CACHE_MAXSIZE", 64)
    _settings.INDICES_DEBUG = env_bool("PXD_INDICES_DEBUG", False)
    _settings.SHAPE_CACHE_MAXSIZE = env_int("PXD_SHAPE_CACHE_MAXSIZE", 128)
    _settings.PREFIX_CACHE_ENABLED = env_bool("PXD_PREFIX_CACHE_ENABLED", True)
    _settings.PREFIX_CACHE_MAXSIZE = env_int("PXD_PREFIX_CACHE_MAXSIZE", 128)
    _settings.PREFIX_CACHE_MAX_VERTS = env_int("PXD_PREFIX_CACHE_MAX_VERTS", 10_000_000) or 0
    _settings.DEBUG_PREFIX_CACHE = env_bool("PXD_DEBUG_PREFIX_CACHE", False)
    _settings.COMPILED_CACHE_MAXSIZE = env_int("PXD_COMPILED_CACHE_MAXSIZE", 128)
    _settings.DEBUG_FONTS = env_bool("PXD_DEBUG_FONTS", False)
    _settings.USE_NUMBA = env_bool("PYX_USE_NUMBA", True)

def get() -> _Settings:
    return _settings

# 初期ロード
reload_from_env()
```

チェックリスト（段階導入）

- [x] Phase 1: settings 追加
  - [x] `src/common/settings.py` を上記案で新規作成（docstring/型/関数含む）。
  - [x] 最小 smoke 実行でインポート/型エラーが無いことを確認（import 時ロード）。

- [x] Phase 2: 最小セット置換
  - [x] `src/common/param_utils.py`（`PXD_PIPELINE_QUANT_STEP`）を settings 参照へ置換。
  - [x] `src/engine/ui/parameters/persistence.py`（同上）を settings 参照へ置換。
  - [x] ruff/black/isort/mypy/対象テストを変更ファイル限定で実行。

- [x] Phase 3: 形状/プレフィックス LRU 置換
  - [x] `src/engine/core/lazy_geometry.py` の `env_int/env_bool` 呼び出しを settings へ置換。
  - [ ] カウンタやデバッグ挙動が不変であることを確認。

- [x] Phase 4: renderer 置換
  - [x] `src/engine/render/renderer.py` の IBO/indices 系環境変数を settings へ置換。
  - [ ] 既定値・フォールバック時の挙動（例外時のデフォルト）が等価であることを確認。

- [x] Phase 5: util/effects 置換
  - [x] `src/util/fonts.py` の `PXD_DEBUG_FONTS` を settings 参照へ。
  - [x] `src/effects/collapse.py` の `PYX_USE_NUMBA` を settings 参照へ。

- [ ] Phase 6: ドキュメント同期/不要変数の整理
  - [ ] `docs/environment_variables.md` 更新（集中管理化済みの注記）。
  - [ ] 未使用: `PXD_IBO_DEBUG`, `PXD_INDICES_DEBUG` の扱いを決定（削除 or 使途追加）。
  - [ ] 文書のみ: `PXD_PIPELINE_CACHE_MAXSIZE` の導入可否を決定（導入 or 記述削除）。

検証ポリシー（編集ファイル優先）
- Lint: `ruff check --fix {changed_files}`
- Format: `black {changed_files} && isort {changed_files}`
- Type: `mypy {changed_files}`
- Test: 変更に関係するテストのみ（例: `pytest -q -k lazy_geometry` など）。

承認後、Phase 1 から着手します。必要があれば優先順位やスコープを調整します。
