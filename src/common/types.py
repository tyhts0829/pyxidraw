"""
どこで: `common` の型定義。
何を: Vec2/Vec3 などの軽量エイリアス（組込みジェネリックで記述）。
なぜ: 依存の少ない場所に配置して循環と分散定義を避けるため。
"""

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


class ObjectRef:
    """任意オブジェクトの参照を保持する軽量ラッパ（ハッシュ可能）。

    - 目的: パイプライン署名生成時に `make_hashable_param` を経由しても、
      実体オブジェクトを失わずに plan 内へ保持するための薄い参照。
    - ハッシュ/等価性は参照先オブジェクトの `id` に基づく（同一性）。
    - `unwrap()` で参照先を取り出す。
    """

    __slots__ = ("obj",)

    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __hash__(self) -> int:  # pragma: no cover - 単純実装
        return hash(id(self.obj))

    def __eq__(self, other: object) -> bool:  # pragma: no cover - 単純実装
        return isinstance(other, ObjectRef) and self.obj is other.obj

    def __repr__(self) -> str:  # pragma: no cover - 表示用
        cname = getattr(self.obj, "__class__", type(self.obj)).__qualname__
        return f"ObjectRef({cname}@{id(self.obj):x})"

    def unwrap(self) -> object:
        return self.obj


__all__ = ["Vec2", "Vec3", "ObjectRef"]
