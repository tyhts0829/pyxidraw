"""
共通キャッシング機能基底クラス
shapes/ と effects/ の両方で使用する統一されたキャッシング機能
"""

import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple


class CacheableBase(ABC):
    """キャッシング機能を持つ基底クラス"""

    def __init__(self):
        self._cache_enabled: bool = True
        self._cache_size: int = 128
        self._cache: Dict[str, Any] = {}

    def enable_cache(self) -> None:
        """キャッシュを有効化"""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """キャッシュを無効化"""
        self._cache_enabled = False

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """キャッシュキーを生成"""
        # パラメータをハッシュ化
        params_str = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(params_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        if not self._cache_enabled:
            return None
        return self._cache.get(cache_key)

    def _store_in_cache(self, cache_key: str, value: Any) -> None:
        """キャッシュに値を保存"""
        if not self._cache_enabled:
            return

        # キャッシュサイズ制限
        if len(self._cache) >= self._cache_size:
            # 最も古いエントリを削除（LRU風）
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = value

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """実際の処理を実行（サブクラスで実装）"""
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """キャッシュ機能付きの実行"""
        if not self._cache_enabled:
            return self._execute(*args, **kwargs)

        cache_key = self._generate_cache_key(*args, **kwargs)
        cached_result = self._get_from_cache(cache_key)

        if cached_result is not None:
            return cached_result

        result = self._execute(*args, **kwargs)
        self._store_in_cache(cache_key, result)
        return result


class LRUCacheable(CacheableBase):
    """LRUキャッシュを使用するバージョン"""

    def __init__(self, maxsize: int = 128):
        super().__init__()
        self._cache_size = maxsize
        self._setup_lru_cache()

    def _setup_lru_cache(self) -> None:
        """LRUキャッシュをセットアップ"""
        self._cached_execute = lru_cache(maxsize=self._cache_size)(self._execute)

    def __call__(self, *args, **kwargs) -> Any:
        """LRUキャッシュ機能付きの実行"""
        if not self._cache_enabled:
            return self._execute(*args, **kwargs)
        return self._cached_execute(*args, **kwargs)

    @property
    def cache_enabled(self) -> bool:
        return self._cache_enabled

    def enable_cache(self) -> None:
        """キャッシュを有効化（LRUキャッシュを再セットアップ）"""
        super().enable_cache()
        if not hasattr(self, "_cached_execute"):
            self._setup_lru_cache()

    def disable_cache(self) -> None:
        """キャッシュを無効化"""
        super().disable_cache()
        if hasattr(self, "_cached_execute"):
            self._cached_execute.cache_clear()
