"""
シェイプレジストリ - 統一されたレジストリシステム
effects/ と対称性を保った @shape デコレータの実装。

改善点:
- 名前省略可: `@shape` や `@shape()` でクラス名から自動推論（例: `Sphere` -> `sphere`）。
  エイリアスはサポートしません（明示名のみ）。
"""

from common.base_registry import BaseRegistry

from .base import BaseShape

# 統一されたレジストリシステム
_shape_registry = BaseRegistry()


def shape(arg: str | type[BaseShape] | None = None):  # type: ignore[name-defined]
    """シェイプをレジストリに登録するデコレータ。

    使い方:
    - 省略名で登録: `@shape` または `@shape()` -> クラス名から推論（`Sphere` -> `sphere`）
    - 明示名で登録: `@shape("sphere")` も可能（必要な場合のみ）
    """
    # @shape のように引数なしで使われたケース
    if callable(arg):
        return _shape_registry.register(None)(arg)  # type: ignore[arg-type]
    # @shape() / @shape("name") のケース
    name = arg if isinstance(arg, str) else None
    return _shape_registry.register(name)


def get_shape(name: str) -> type[BaseShape]:
    """登録されたシェイプクラスを取得。

    引数:
        name: シェイプ名

    返り値:
        シェイプクラス

    例外:
        KeyError: シェイプが登録されていない場合
    """
    return _shape_registry.get(name)


def list_shapes() -> list[str]:
    """登録されているシェイプの一覧を取得。

    返り値:
        シェイプ名のリスト
    """
    return _shape_registry.list_all()


def is_shape_registered(name: str) -> bool:
    """シェイプが登録されているかチェック。

    引数:
        name: シェイプ名。

    返り値:
        登録されている場合は True。
    """
    return _shape_registry.is_registered(name)


def clear_registry():
    """レジストリをクリア（テスト用）。"""
    _shape_registry.clear()


def get_registry() -> BaseRegistry:
    """レジストリインスタンスを取得（ファクトリクラス用）。

    返り値:
        レジストリインスタンス。
    """
    return _shape_registry
