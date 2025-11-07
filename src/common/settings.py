"""
どこで: `common.settings`
何を: プロジェクトの環境変数を型付きで一元管理し、起動時に読み込む。
なぜ: `os.getenv` の散在を解消し、既定値/型の一貫性とテスト容易性を高めるため。
"""

from __future__ import annotations

from dataclasses import dataclass

from .env import env_bool, env_int


@dataclass
class _Settings:
    # 量子化
    PIPELINE_QUANT_STEP: float = 1e-6

    # Renderer / indices
    IBO_FREEZE_ENABLED: bool = True
    IBO_DEBUG: bool = False
    INDICES_CACHE_ENABLED: bool = True
    INDICES_CACHE_MAXSIZE: int | None = 64
    INDICES_DEBUG: bool = False

    # LazyGeometry caches
    SHAPE_CACHE_MAXSIZE: int | None = 128
    PREFIX_CACHE_ENABLED: bool = True
    PREFIX_CACHE_MAXSIZE: int | None = 128
    PREFIX_CACHE_MAX_VERTS: int = 10_000_000
    DEBUG_PREFIX_CACHE: bool = False

    # Effects / API
    COMPILED_CACHE_MAXSIZE: int | None = 128

    # Misc
    DEBUG_FONTS: bool = False
    USE_NUMBA: bool = True


_settings = _Settings()


def reload_from_env() -> None:
    """環境変数から設定を再読込。

    - bool は `env_bool`、int は `env_int` を使用。
    - 一部は下限丸めやフォールバックを適用。
    """

    # 量子化（float は個別処理: env ヘルパは提供していない）
    try:
        import os

        raw = os.getenv("PXD_PIPELINE_QUANT_STEP")
        _settings.PIPELINE_QUANT_STEP = float(raw) if raw is not None else 1e-6
    except Exception:
        _settings.PIPELINE_QUANT_STEP = 1e-6

    # Renderer / indices
    _settings.IBO_FREEZE_ENABLED = env_bool("PXD_IBO_FREEZE_ENABLED", True)
    _settings.IBO_DEBUG = env_bool("PXD_IBO_DEBUG", False)
    _settings.INDICES_CACHE_ENABLED = env_bool("PXD_INDICES_CACHE_ENABLED", True)
    _settings.INDICES_CACHE_MAXSIZE = env_int("PXD_INDICES_CACHE_MAXSIZE", 64)
    if _settings.INDICES_CACHE_MAXSIZE is not None and _settings.INDICES_CACHE_MAXSIZE < 0:
        _settings.INDICES_CACHE_MAXSIZE = 0
    _settings.INDICES_DEBUG = env_bool("PXD_INDICES_DEBUG", False)

    # LazyGeometry caches（下限丸め）
    _settings.SHAPE_CACHE_MAXSIZE = env_int("PXD_SHAPE_CACHE_MAXSIZE", 128, min_value=0)
    _settings.PREFIX_CACHE_ENABLED = env_bool("PXD_PREFIX_CACHE_ENABLED", True)
    _settings.PREFIX_CACHE_MAXSIZE = env_int("PXD_PREFIX_CACHE_MAXSIZE", 128, min_value=0)
    _settings.PREFIX_CACHE_MAX_VERTS = (
        env_int("PXD_PREFIX_CACHE_MAX_VERTS", 10_000_000, min_value=0) or 0
    )
    _settings.DEBUG_PREFIX_CACHE = env_bool("PXD_DEBUG_PREFIX_CACHE", False)

    # Effects / API
    _settings.COMPILED_CACHE_MAXSIZE = env_int("PXD_COMPILED_CACHE_MAXSIZE", 128)
    if _settings.COMPILED_CACHE_MAXSIZE is not None and _settings.COMPILED_CACHE_MAXSIZE < 0:
        _settings.COMPILED_CACHE_MAXSIZE = 0

    # Misc
    _settings.DEBUG_FONTS = env_bool("PXD_DEBUG_FONTS", False)
    _settings.USE_NUMBA = env_bool("PYX_USE_NUMBA", True)


def get() -> _Settings:
    """現在の設定スナップショットを返す。"""
    return _settings


# 初期ロード
reload_from_env()


__all__ = ["get", "reload_from_env", "_Settings"]
