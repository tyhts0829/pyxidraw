"""
G - 形状ファクトリ（高レベル API エントリ）

概要:
- ユーザコードからは `from api import G` として利用し、`G.sphere(...)` のような
  「関数的」な呼び出しで `Geometry` を生成する。
- 実体はレジストリ（`shapes.registry`）に登録済みの各シェイプクラス（`BaseShape` 派生）を
  動的ディスパッチで解決し、インスタンス化→`generate(**params)` を呼ぶ薄いファサード。
- 生成結果は必ず `engine.core.geometry.Geometry`（以下 Geometry）に揃える。
  旧実装の互換として、`(coords, offsets)` 等の「線分リスト」形式が返る場合は
  `Geometry.from_lines(...)` で包む。

プロジェクト内での役割（architecture.md 要約）:
- 形状生成の単一入口: `G` が「どのシェイプをどう作るか」を一手に引き受ける。
- 責務の境界: "生成" は `G`/各 `Shape`、"変換" は `Geometry.translate/scale/rotate/concat`、
  "加工" は `E.pipeline` に委譲。シェイプ側での変換パラメータは互換のため残るが推奨しない。
- キャッシュの責務: 形状生成の LRU は本モジュール（`ShapeFactory._cached_shape`）に集約
  （ADR 0011）。`BaseShape` 側の LRU は既定で無効化（必要時のみ opt-in）。

設計上のポイント:
- 動的ディスパッチ: メタクラス `ShapeFactoryMeta` とインスタンス `__getattr__` の双方で
  `G.sphere(...)`/`ShapeFactory.sphere(...)` をサポート（クラス/インスタンスどちらからでも可）。
- 高速化と再現性: `_params_to_tuple()` でパラメータを「順序安定かつハッシュ可能」へ正規化し、
  `functools.lru_cache` による 1 プロセス内 LRU キャッシュを掛ける。
- 例外方針: 未登録名は `AttributeError`（動的属性の一貫性）、生成器側の失敗は各シェイプが責任。

使用例:
    from api import G, E

    # 形状の生成と Geometry 変換
    g = G.sphere(subdivisions=0.5).scale(100, 100, 100).translate(100, 100, 0)

    # パイプラインの適用（加工は E.pipeline に委譲）
    pipe = (E.pipeline.displace(amplitude_mm=0.2)
                        .fill(density=0.5)
                        .build())
    g2 = pipe(g)

運用メモ:
- 本モジュールの LRU は `functools.lru_cache(maxsize=128)` 固定（環境変数では切り替えない）。
  `BaseShape` の LRU（無効が既定）は `PXD_CACHE_DISABLED`/`PXD_CACHE_MAXSIZE` で調整可能（共通基盤）。
- スレッドセーフティ: CPython の `lru_cache` は内部ロックを持つため、並列呼び出しでも基本安全。
  ただし各シェイプの `generate` は純粋関数（副作用なし）であることを前提にする。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Iterable, Callable, Any, cast
import hashlib

import numpy as np

# レジストリ登録の副作用を発火させるため、shapes パッケージを 1 度だけ import すれば十分
import shapes  # noqa: F401  (登録目的の副作用)
from engine.core.geometry import Geometry
from numpy.typing import NDArray
from .shape_registry import get_shape_generator, list_registered_shapes


ParamsTuple = tuple[tuple[str, object], ...]


class ShapeFactoryMeta(type):
    """ShapeFactory のメタクラス。

    目的:
    - クラスレベル（`ShapeFactory.sphere(...)` のような呼び出し）でも動的属性を解決し、
      レジストリにあるシェイプ名を関数として露出する。
    - 返す関数は `ShapeFactory._cached_shape(...)` を経由し、LRU キャッシュの恩恵を受ける。
    """

    def __getattr__(cls, name: str) -> Callable[..., Geometry]:
        """クラスレベルでの動的属性アクセス。

        指定された `name` がシェイプレジストリに存在する場合、
        `shape_method(**params) -> Geometry` を返す。存在しない場合は `AttributeError`。
        """
        try:
            # 存在確認のみ（実体の生成は _cached_shape に委譲）
            get_shape_generator(name)

            # 動的にクラスメソッドを生成
            def shape_method(**params: object) -> Geometry:
                params_tuple = cls._params_to_tuple(**params)
                # サブクラス差し替えを可能にするため cls 経由で呼び出す
                return cls._cached_shape(name, params_tuple)

            return shape_method
        except ValueError:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")


class ShapeFactory(metaclass=ShapeFactoryMeta):
    """高性能キャッシュ付き形状ファクトリ（`G` の実体）。

    責務:
    - 形状名→生成関数の動的ディスパッチ
    - 生成結果の型統一（常に `Geometry` を返す）
    - 形状生成結果の LRU キャッシュ（maxsize=128）

    使い方:
        from api import G
        g1 = G.sphere(subdivisions=0.5)
        g2 = G.polygon(n_sides=6)
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_shape(shape_name: str, params_tuple: ParamsTuple) -> Geometry:
        """登録シェイプを解決・生成し、結果を LRU キャッシュ。

        Args:
            shape_name: シェイプ名（レジストリキー）
            params_tuple: `_params_to_tuple(**params)` が返すハッシュ可能タプル

        Returns:
            Geometry: 生成された形状（必要に応じて `from_lines` でラップ）

        Note:
            - キャッシュキーは `(shape_name, params_tuple)`。
            - 生成器（`BaseShape.generate`）が Geometry 以外の互換形式を返すケースを吸収する。
        """
        params_dict = dict(params_tuple)

        # レジストリから形状クラスを取得し、インスタンスを生成
        shape_cls = get_shape_generator(shape_name)
        instance = shape_cls()
        data = instance.generate(**params_dict)
        # 生成器は Geometry を返す前提
        if isinstance(data, Geometry):
            return data
        return Geometry.from_lines(data)

    @staticmethod
    def _params_to_tuple(**params: object) -> ParamsTuple:
        """パラメータ辞書を「順序安定・ハッシュ可能」なタプルへ正規化。

        - dict は key でソートし `(key, value)` のタプル列に。
        - list/tuple は要素ごとに再帰変換。
        - NumPy 配列は `("nd", shape, dtype, blake2b-128 digest)` という短い指紋に変換。
          `dtype` と `shape` を含めるため、単なるフラット化による情報欠落を避けられる。

        注意:
        - オブジェクト配列（`dtype=object`）は `tolist()` で中身を列挙して再帰的に正規化する。
        - 未対応/非ハッシュ可能オブジェクトは安全側フォールバックで
          `('obj', <qualified_class_name>, id(obj))` のタプルに変換する（同一インスタンスのみヒット）。
        """

        # ネストした値も含めて再帰的にソート・変換
        def _key_for_sorting_object_key(k: object) -> str:
            return f"{type(k).__name__}:{repr(k)}"

        def _sort_key_item(kv: tuple[str, object]) -> str:
            return _key_for_sorting_object_key(kv[0])

        def make_hashable(obj: object) -> object:
            # dict: キー型が混在しても安定比較できるよう型名+reprでソート
            if isinstance(obj, dict):
                items_iter = cast(Iterable[tuple[object, object]], obj.items())
                items = sorted(items_iter, key=lambda kv: _key_for_sorting_object_key(kv[0]))
                return tuple((k, make_hashable(v)) for k, v in items)

            # シーケンス: 再帰的にタプル化
            if isinstance(obj, (list, tuple)):
                return tuple(make_hashable(item) for item in obj)

            # NumPy スカラーは Python 組込みへ
            if isinstance(obj, np.generic):
                return obj.item()

            # NumPy 配列: dtype/shape を含む指紋化で巨大キー化と衝突を回避
            if isinstance(obj, np.ndarray):
                if obj.dtype.kind == 'O':
                    # オブジェクト配列は素直に中身を列挙
                    return ('nd_obj', tuple(make_hashable(x) for x in obj.tolist()))
                arr = np.ascontiguousarray(obj)
                h = hashlib.blake2b(digest_size=16)
                h.update(arr.view(np.uint8).tobytes())
                return ('nd', arr.shape, str(arr.dtype), h.digest())

            # set/frozenset: 並びに依らず安定化
            if isinstance(obj, (set, frozenset)):
                return (
                    'set',
                    tuple(
                        sorted(
                            (make_hashable(x) for x in obj),
                            key=_key_for_sorting_object_key,
                        )
                    ),
                )

            # bytes/bytearray
            if isinstance(obj, (bytes, bytearray)):
                return bytes(obj)

            # ハッシュ可能であればそのまま返す
            try:
                hash(obj)  # type: ignore[arg-type]
                return obj
            except Exception:
                # 非ハッシュ可能（例: ユーザ定義の __hash__=None）の場合でも
                # lru_cache のキー化で落ちないように安全側フォールバック。
                # 内容同値性よりも「プロセス内で同一インスタンスを同一視」を優先する。
                cls_name = getattr(obj, "__class__", type(obj)).__qualname__
                return ('obj', cls_name, id(obj))

        params_dict: dict[str, object] = dict(params)
        items = sorted(params_dict.items(), key=_sort_key_item)
        return tuple((k, make_hashable(v)) for k, v in items)

    # === レジストリベース形状生成（メタクラスによる動的アクセス） ===

    # === ユーザー拡張 ===

    @staticmethod
    def from_lines(lines: Iterable[NDArray[Any]]) -> Geometry:
        """線分集合（ポリライン列）から `Geometry` を構築する補助。

        Args:
            lines: `[(N_i, 3) ndarray]` を要素とするイテラブル

        Returns:
            Geometry: `coords`/`offsets` を持つ統一 Geometry
        """
        return Geometry.from_lines(lines)

    @staticmethod
    def empty() -> Geometry:
        """空の `Geometry`（頂点ゼロ）を返すユーティリティ。"""
        return Geometry.from_lines([])

    def __getattr__(self, name: str) -> Callable[..., Geometry]:
        """インスタンスレベルでの動的属性アクセス。

        `G.<name>(**params) -> Geometry` を提供。クラスレベルと同じロジックで
        レジストリ解決とキャッシュを行う。
        """
        # レジストリに登録されているか確認
        try:
            get_shape_generator(name)

            # 動的にメソッドを生成
            def shape_method(**params: object) -> Geometry:
                params_tuple = self._params_to_tuple(**params)
                return self._cached_shape(name, params_tuple)

            return shape_method
        except ValueError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def list_shapes(cls) -> List[str]:
        """利用可能な形状（レジストリ登録済み名）の一覧を返す。"""
        return list_registered_shapes()

    # === キャッシュ管理 ===

    @classmethod
    def clear_cache(cls) -> None:
        """形状生成結果の LRU キャッシュをクリア（デバッグ/メモリ回収用）。"""
        cls._cached_shape.cache_clear()

    @classmethod
    def cache_info(cls) -> object:
        """`functools.lru_cache` の統計情報を取得（ヒット率等の観察に）。"""
        return cls._cached_shape.cache_info()

    # 補完体験向上: dir(G) で登録シェイプ名を出す
    def __dir__(self) -> list[str]:
        try:
            base = set(object.__dir__(self))
        except Exception:
            base = set()
        return sorted(base.union(list_registered_shapes()))


# シングルトンインスタンス（`from api import G` で公開）
G = ShapeFactory()
