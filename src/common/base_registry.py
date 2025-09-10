"""
共通レジストリ基底クラス
shapes/ と effects/ の両方で使用する統一されたレジストリシステム
"""

import re
from abc import ABC
from typing import Any, Callable


class BaseRegistry(ABC):
    """レジストリの基底クラス。

    - 文字列キーは正規化されます（大文字小文字・キャメル→スネークを吸収）。
    - デコレータは名前省略可。省略時はクラス/関数名から自動推論します。
    """

    def __init__(self):
        # 登録対象の型は統一せず Any とする（関数/クラスの双方を許容）。
        # これにより shapes/effects の API を同一ポリシーで運用できる。
        self._registry: dict[str, Any] = {}

    # === 内部ユーティリティ ===
    @staticmethod
    def _camel_to_snake(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()

    @classmethod
    def _normalize_key(cls, name: str) -> str:
        """レジストリキーの正規化（例: "MyEffect" -> "my_effect"）。"""
        if not isinstance(name, str):
            raise TypeError("レジストリキーは str である必要があります")
        if not name:
            raise ValueError("レジストリキーは空であってはなりません")
        # 既にスネークケースならそのまま、キャメルケースなら変換
        # アンダースコアやハイフンを含む場合も小文字化して扱う
        name = name.replace("-", "_")
        # 大文字を含む場合のみキャメル→スネーク変換
        return cls._camel_to_snake(name) if any(c.isupper() for c in name) else name.lower()

    def register(self, name: str | None = None) -> Callable:
        """クラス/関数をレジストリに登録するデコレータ。"""

        def decorator(obj: Any) -> Any:
            key = self._normalize_key(name) if name else self._normalize_key(obj.__name__)
            if key in self._registry and self._registry[key] is not obj:
                raise ValueError(f"'{key}' は既に登録されています")
            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        """登録されたクラス/関数を取得。"""
        key = self._normalize_key(name)
        if key not in self._registry:
            raise KeyError(f"'{name}' は登録されていません")
        return self._registry[key]

    def list_all(self) -> list[str]:
        """登録されているすべての名前を取得（未ソート）。"""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """指定された名前が登録されているかチェック"""
        return self._normalize_key(name) in self._registry

    def unregister(self, name: str) -> None:
        """レジストリから削除（名前が存在しない場合は無視）。"""
        key = self._normalize_key(name)
        if key in self._registry:
            del self._registry[key]

    def clear(self) -> None:
        """レジストリをクリア"""
        self._registry.clear()

    @property
    def registry(self) -> dict[str, Any]:
        """レジストリの読み取り専用アクセス"""
        return self._registry.copy()

    # 後方互換のためのリネーム機能は提供しない（明示的な再登録を推奨）


class CacheableRegistry(BaseRegistry):
    """キャッシング機能付きレジストリ"""

    def __init__(self):
        super().__init__()
        self._instance_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}

    def get_instance(self, name: str, **kwargs) -> Any:
        """インスタンスを取得（キャッシュ機能付き）"""
        cache_key = (name, tuple(sorted(kwargs.items())))

        if cache_key not in self._instance_cache:
            cls = self.get(name)
            instance = cls(**kwargs)
            self._instance_cache[cache_key] = instance

        return self._instance_cache[cache_key]

    def clear_instance_cache(self) -> None:
        """インスタンスキャッシュをクリア"""
        self._instance_cache.clear()
