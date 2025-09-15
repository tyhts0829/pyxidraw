"""
api.shapes — 形状生成の入口（高レベル API）

Notes
-----
- 利用者は `from api import G` として、`G.sphere(...)` のように関数的に呼び出して
  `Geometry` を生成する。
- 実体はレジストリ（`shapes.registry`）に登録済みの shape 関数を解決し、
  `fn(**params)` を直接呼ぶ薄いファサード。
- 生成結果は常に `engine.core.geometry.Geometry`（以下 Geometry）。Geometry 以外を返す
  シェイプ関数がある場合は「ポリライン列（list/ndarray の列）」のみ `Geometry.from_lines(...)`
  で受け付ける。旧タプル形式 `(coords, offsets)` は非サポート。
- LRU は `functools.lru_cache(maxsize=128)` 固定。
- CPython の `lru_cache` は内部ロックを持ち、並列呼び出しでも基本安全。
  各シェイプの `generate` は純粋関数（副作用なし）であることを前提にする。

Design
------
- 単一入口: 形状生成は `G` が一手に引き受ける。
- 責務境界: 生成は `G`/各 shape 関数、変換は `Geometry`、加工は `E.pipeline`。
- キャッシュ: 形状生成結果の LRU（maxsize=128）は本モジュールに集約。
- 動的ディスパッチ: メタクラス `ShapesAPIMeta` とインスタンス `__getattr__` の双方で
  `G.sphere(...)`/`ShapesAPI.sphere(...)` をサポート。
- 再現性と性能: `_params_to_tuple()` でパラメータを決定的・ハッシュ可能に正規化し、
  プロセス内 LRU を適用。
- 例外方針: 未登録名は `AttributeError`。生成器側の失敗は各シェイプが責任。

Examples
--------
    from api import G, E

    g = (G.sphere(subdivisions=0.5)
           .scale(100, 100, 100)
           .translate(100, 100, 0))

    pipe = (E.pipeline
              .displace(amplitude_mm=0.2)
              .fill(density=0.5)
              .build())
    g2 = pipe(g)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Iterable

from numpy.typing import NDArray

# レジストリ登録の副作用を発火させるため、shapes パッケージを 1 度だけ import すれば十分
import shapes  # noqa: F401  (登録目的の副作用)
from common.param_utils import params_to_tuple as _params_to_tuple
from engine.core.geometry import Geometry
from shapes.registry import get_shape as get_shape_generator
from shapes.registry import is_shape_registered  # ガード付きメモ化用（登録状態の即時反映）
from shapes.registry import list_shapes as list_registered_shapes

# 形状生成のキャッシュ鍵として用いる正規化済みパラメータ列。
# プロジェクト規約に合わせ、値の型は `Any` を用いる。
ParamsTuple = tuple[tuple[str, Any], ...]


class ShapesAPIMeta(type):
    """ShapesAPI のメタクラス。

    目的:
    - クラスレベル（`ShapesAPI.sphere(...)` のような呼び出し）でも動的属性を解決し、
      レジストリにあるシェイプ名を関数として露出する。
    - 返す関数は `ShapesAPI._cached_shape(...)` を経由し、LRU キャッシュの恩恵を受ける。
    """

    def __getattr__(cls, name: str) -> Callable[..., Geometry]:
        """クラスレベルでの動的属性アクセス。

        指定された `name` がシェイプレジストリに存在する場合、
        `shape_method(**params) -> Geometry` を返す。
        存在しない場合は `AttributeError`。
        """
        try:
            # 存在確認のみ（実体の生成は _cached_shape に委譲）
            _ = get_shape_generator(name)

            # 動的にクラスメソッドを生成し、クラス属性としてメモ化する。
            # 注意（重要）:
            # - この関数は staticmethod としてクラスにバインドする。これによりインスタンスから
            #   参照しても `self` が暗黙に注入されず、`G.sphere(...)`/`ShapesAPI.sphere(...)`
            #   の双方で同一実装を安全に共有できる。
            # - ランタイムでの登録解除に対応するため、呼び出し時に `is_shape_registered(name)` を
            #   検査するガードを設置。未登録ならクラス属性を削除して `AttributeError` を投げ、
            #   動的属性の整合性（未登録=属性なし）を保つ。（メモ化の副作用を相殺するための安全策）
            def _shape_method_impl(**params: Any) -> Geometry:
                if not is_shape_registered(name):
                    # 登録が解除されていた場合はメモ化を破棄して AttributeError へ整合
                    try:
                        delattr(cls, name)
                    except Exception:
                        pass
                    raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")
                params_tuple = cls._params_to_tuple(**params)
                return cls._cached_shape(name, params_tuple)

            # ヘルプ/補完のためにメタデータを持たせる
            _shape_method_impl.__name__ = name
            _shape_method_impl.__qualname__ = f"{cls.__name__}.{name}"
            _doc = None  # 関数ベースへ移行後は docstring の転写は省略（任意で将来対応）
            if _doc:
                _shape_method_impl.__doc__ = _doc

            # クラスへ staticmethod としてメモ化
            setattr(cls, name, staticmethod(_shape_method_impl))
            # 実際に返すのはクラス属性（staticmethod 解決後の関数オブジェクト）
            return getattr(cls, name)
        except Exception:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")


class ShapesAPI(metaclass=ShapesAPIMeta):
    """高性能キャッシュ付き形状 API（`G` の実体）。

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

        引数:
            shape_name: シェイプ名（レジストリキー）。
            params_tuple: `_params_to_tuple(**params)` が返すハッシュ可能タプル。

        返り値:
            生成された形状（必要に応じて `from_lines` でラップ）を表す `Geometry`。

        Note:
            - キャッシュキーは `(shape_name, params_tuple)`。
            - 生成器（shape 関数）が Geometry 以外の互換形式を返すケースを吸収する。
        """
        params_dict = dict(params_tuple)

        # レジストリから形状関数を取得して呼び出す
        fn = get_shape_generator(shape_name)
        data = fn(**params_dict)
        # 生成器は Geometry を返す前提
        if isinstance(data, Geometry):
            return data
        return Geometry.from_lines(data)

    @staticmethod
    def _params_to_tuple(**params: Any) -> ParamsTuple:
        """パラメータ辞書を「順序安定・ハッシュ可能」なタプルへ正規化（共通実装）。"""
        return _params_to_tuple(dict(params))

    # === レジストリベース形状生成（メタクラスによる動的アクセス） ===

    # === ユーザー拡張 ===

    @staticmethod
    def from_lines(lines: Iterable[NDArray[Any]]) -> Geometry:
        """線分集合（ポリライン列）から `Geometry` を構築する補助。

        引数:
            lines: `[(N_i, 3) ndarray]` を要素とするイテラブル。

        返り値:
            `coords` と `offsets` を持つ統一 `Geometry`。
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
        # クラス側の __getattr__ に委譲してメモ化を行い、取得する。
        # 未登録名の場合はクラス側が AttributeError を投げる。
        try:
            return getattr(type(self), name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def list_shapes(cls) -> list[str]:
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
G = ShapesAPI()
