"""
どこで: `api.shapes`（形状生成の高レベル API）。
何を: 登録済み shape 関数を解決して `Geometry` を返す薄いファサード（LRU キャッシュ付き）。
なぜ: 生成（shape）と加工（effects）を分離しつつ、関数的に扱える統一入口を提供するため。

api.shapes — 形状生成の入口（高レベル API）

Notes
-----
- 利用者は `from api import G` として、`G.sphere(...)` のように関数的に呼び出して
  `Geometry` を生成する。
- 実体はレジストリ（`shapes.registry`）に登録済みの shape 関数を解決し、
  `fn(**params)` を直接呼ぶ薄いファサード。
- UI/ランタイム層（`engine.ui.parameters`）経由でも、本モジュールは各シェイプ関数が
  期待する実レンジの値をそのまま受け取り、関数へ引き渡す。
  直接呼び出す場合も同様に、各シェイプ関数が期待する単位系で値を渡す。
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
- 動的ディスパッチ: インスタンス `__getattr__` で遅延解決し、`G.sphere(...)` の形で提供。
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

# レジストリ登録の副作用を発火させるため、shapes パッケージを 1 度だけ import すれば十分
import shapes  # noqa: F401  (登録目的の副作用)
from common.param_utils import params_to_tuple as _params_to_tuple
from common.param_utils import signature_tuple as _signature_tuple
from engine.core.geometry import Geometry, LineLike
from engine.ui.parameters import get_active_runtime
from engine.ui.parameters.runtime import resolve_without_runtime
from shapes.registry import get_shape as get_shape_generator
from shapes.registry import is_shape_registered  # ガード付きメモ化用（登録状態の即時反映）
from shapes.registry import list_shapes as list_registered_shapes

# 形状生成のキャッシュ鍵として用いる正規化済みパラメータ列。
# プロジェクト規約に合わせ、値の型は `Any` を用いる。
ParamsTuple = tuple[tuple[str, Any], ...]


class ShapesAPI:
    """高性能キャッシュ付き形状 API（`G` の実体）。

    責務:
    - 形状名→生成関数の動的ディスパッチ（インスタンス属性で遅延解決）
    - 生成結果の型統一（常に `Geometry` を返す）
    - 形状生成結果の LRU キャッシュ（maxsize=128）

    Notes
    -----
    UI/ランタイム層経由でも実値をそのまま扱う。
    本 API を直接呼ぶ場合は各シェイプが期待する単位系（例: ミリメートル、度数）で
    値を指定すること。

    使い方:
        from api import G
        g1 = G.sphere(subdivisions=0.5)
        g2 = G.polygon(n_sides=6)
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_shape(shape_name: str, params_tuple: ParamsTuple) -> Geometry:
        """登録シェイプを生成して LRU に保存する。

        Parameters
        ----------
        shape_name : str
            形状名（レジストリ登録済みキー）。
        params_tuple : ParamsTuple
            パラメータ辞書を順序安定・ハッシュ可能に正規化したタプル。

        Returns
        -------
        Geometry
            生成された `Geometry`。

        Notes
        -----
        - キャッシュキーは `(shape_name, params_tuple)`。
        - 例外は下層からそのまま伝播。
        """
        params_dict = dict(params_tuple)
        return ShapesAPI._generate_shape_resolved(shape_name, params_dict)

    @staticmethod
    def _generate_shape_resolved(shape_name: str, params_dict: dict[str, Any]) -> Geometry:
        """Runtime 介在なしで直接シェイプ関数を実行し Geometry を返す。"""
        fn = get_shape_generator(shape_name)
        data = fn(**params_dict)
        if isinstance(data, Geometry):
            return data
        return Geometry.from_lines(data)

    # 量子化ユーティリティは common.param_utils.signature_tuple を使用

    @staticmethod
    def _generate_shape(shape_name: str, params_dict: dict[str, Any]) -> Geometry:
        """形状生成関数を解決して実行し、`Geometry` を返す。

        Parameters
        ----------
        shape_name : str
            形状名（レジストリ登録済みキー）。
        params_dict : dict[str, Any]
            形状関数へ渡す実レンジのパラメータ（正規化値ではない）。

        Returns
        -------
        Geometry
            生成結果。戻り値が `Geometry` 以外（ポリライン列）の場合は
            `Geometry.from_lines(...)` で包んで返す。

        Notes
        -----
        - ランタイムが有効な場合は `engine.ui.parameters` による事前解決を行う。
        - 例外は各シェイプ実装からそのまま伝播。
        """
        runtime = get_active_runtime()
        fn = get_shape_generator(shape_name)
        if runtime is not None:
            params_dict = dict(runtime.before_shape_call(shape_name, fn, params_dict))
        else:
            params_dict = dict(
                resolve_without_runtime(
                    scope="shape",
                    name=shape_name,
                    fn=fn,
                    params=params_dict,
                    index=0,
                )
            )

        data = fn(**params_dict)
        if isinstance(data, Geometry):
            return data
        return Geometry.from_lines(data)

    @staticmethod
    def _params_to_tuple(**params: Any) -> ParamsTuple:
        """ハッシュ可能なタプルへパラメータを正規化する。

        Parameters
        ----------
        **params : Any
            形状関数へ渡される任意のパラメータ。

        Returns
        -------
        ParamsTuple
            キー昇順かつ不変要素に正規化されたタプル。

        Notes
        -----
        キャッシュキーの安定化と等価性の保証が目的。
        """
        return _params_to_tuple(dict(params))

    def _build_shape_method(self, name: str) -> Callable[..., Geometry]:
        """レジストリ名から `G.<name>(**params)` を構築する。

        Parameters
        ----------
        name : str
            形状名（レジストリ登録済みキー）。

        Returns
        -------
        Callable[..., Geometry]
            `G.<name>(**params) -> Geometry` となる呼び出し可能。

        Notes
        -----
        - ランタイム無効時は LRU キャッシュを使用。
        - 登録が外れた場合は `AttributeError` を送出。
        """

        def _shape_method(**params: Any) -> Geometry:
            if not is_shape_registered(name):
                # 登録解除と整合を取るため、キャッシュ済みの属性を破棄して AttributeError を送出
                self.__dict__.pop(name, None)
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
            runtime = get_active_runtime()
            if runtime is not None:
                # Runtime 介在時も LRU を有効化するため、解決後の最終値（量子化後）で鍵を作る
                fn = get_shape_generator(name)
                resolved = dict(runtime.before_shape_call(name, fn, dict(params)))
                meta = getattr(fn, "__param_meta__", {}) or {}
                params_tuple = _signature_tuple(resolved, meta)
                return self._cached_shape(name, params_tuple)
            params_tuple = self._params_to_tuple(**params)
            return self._cached_shape(name, params_tuple)

        _shape_method.__name__ = name
        _shape_method.__qualname__ = f"{self.__class__.__name__}.{name}"
        return _shape_method

    # === ユーザー拡張 ===

    @staticmethod
    def from_lines(lines: Iterable[LineLike]) -> Geometry:
        """線分集合（ポリライン列）から `Geometry` を構築する。

        Parameters
        ----------
        lines : Iterable[LineLike]
            `list`/`tuple`/`ndarray` の混在を許容するポリライン列。

        Returns
        -------
        Geometry
            `coords` と `offsets` を持つ統一 `Geometry`。
        """
        return Geometry.from_lines(lines)

    @staticmethod
    def empty() -> Geometry:
        """空の `Geometry`（頂点ゼロ）を返す。

        Returns
        -------
        Geometry
            要素を持たない `Geometry`。
        """
        return Geometry.from_lines([])

    def __getattr__(self, name: str) -> Callable[..., Geometry]:
        """レジストリに基づき `G.<name>` を遅延生成する。

        Parameters
        ----------
        name : str
            形状名（レジストリ登録済みキー）。

        Returns
        -------
        Callable[..., Geometry]
            `G.<name>(**params) -> Geometry` を返す呼び出し可能。

        Raises
        ------
        AttributeError
            未登録名を指定した場合。
        """
        if not is_shape_registered(name):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        method = self._build_shape_method(name)
        self.__dict__[name] = method
        return method

    @classmethod
    def list_shapes(cls) -> list[str]:
        """利用可能な形状名の一覧を返す。

        Returns
        -------
        list[str]
            レジストリに登録済みの形状名の一覧。
        """
        return list_registered_shapes()

    # === キャッシュ管理 ===

    @classmethod
    def clear_cache(cls) -> None:
        """形状生成結果の LRU キャッシュをクリアする。"""
        cls._cached_shape.cache_clear()

    @classmethod
    def cache_info(cls) -> dict[str, int]:
        """LRU キャッシュの統計情報を取得する（辞書形式）。"""
        info = cls._cached_shape.cache_info()
        # functools._CacheInfo(hits, misses, maxsize, currsize)
        return {
            "hits": getattr(info, "hits", 0),
            "misses": getattr(info, "misses", 0),
            "maxsize": getattr(info, "maxsize", 0),
            "size": getattr(info, "currsize", 0),
        }

    # 補完体験向上: dir(G) で登録シェイプ名を出す
    def __dir__(self) -> list[str]:
        """標準の候補にレジストリ登録名を加えて返す。

        Returns
        -------
        list[str]
            通常の属性名に加え、登録シェイプ名を含む一覧。
        """
        try:
            base = set(object.__dir__(self))
        except Exception:
            base = set()
        return sorted(base.union(list_registered_shapes()))


# シングルトンインスタンス（`from api import G` で公開）
G = ShapesAPI()
