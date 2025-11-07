"""
どこで: `api.shapes`（形状生成の高レベル API）。
何を: 登録済み shape 関数を解決して `LazyGeometry` を返す薄いファサード（既定で遅延）。
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
- 再現性と性能: `params_signature()` でパラメータを量子化→ハッシュ可能に正規化し、
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

from collections import OrderedDict
from typing import Any, Callable, Iterable

# レジストリ登録の副作用を発火させるため、shapes パッケージを 1 度だけ import すれば十分
import shapes  # noqa: F401  (登録目的の副作用)
from common.param_utils import params_signature as _params_signature
from engine.core.geometry import Geometry, LineLike
from engine.core.lazy_geometry import LazyGeometry
from engine.ui.parameters import get_active_runtime
from shapes.registry import get_shape as get_shape_generator
from shapes.registry import is_shape_registered  # ガード付きメモ化用（登録状態の即時反映）
from shapes.registry import list_shapes as list_registered_shapes

# 形状生成のキャッシュ鍵として用いる正規化済みパラメータ列。
# プロジェクト規約に合わせ、値の型は `Any` を用いる。
ParamsTuple = tuple[tuple[str, Any], ...]

# --- Lightweight spec cache (hits/misses only, no Geometry) ------------------
_SPEC_CACHE_MAXSIZE = 256
_spec_seen: "OrderedDict[tuple[str, ParamsTuple], None]" = OrderedDict()
_spec_hits = 0
_spec_misses = 0


def _record_spec(shape_name: str, params_tuple: ParamsTuple) -> None:
    global _spec_hits, _spec_misses
    key = (shape_name, params_tuple)
    if key in _spec_seen:
        # move to end (recent)
        _spec_seen.pop(key, None)
        _spec_seen[key] = None
        _spec_hits += 1
        return
    _spec_seen[key] = None
    _spec_misses += 1
    if len(_spec_seen) > _SPEC_CACHE_MAXSIZE:
        try:
            _spec_seen.popitem(last=False)
        except Exception:
            pass


class ShapesAPI:
    """高性能キャッシュ付き形状 API（`G` の実体）。

    責務:
    - 形状名→生成関数の動的ディスパッチ（インスタンス属性で遅延解決）
    - 生成結果の型統一（`LazyGeometry` を返す）

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
    def _lazy_shape(shape_name: str, params_tuple: ParamsTuple) -> LazyGeometry:
        """登録シェイプを spec として Lazy に構築する。"""
        params_dict = dict(params_tuple)
        fn = get_shape_generator(shape_name)
        impl = getattr(fn, "__shape_impl__", fn)
        return LazyGeometry(base_kind="shape", base_payload=(impl, params_dict))

    @staticmethod
    def _generate_shape_resolved(shape_name: str, params_dict: dict[str, Any]) -> Geometry:
        """Runtime 介在なしで直接シェイプ関数を実行し Geometry を返す。"""
        fn = get_shape_generator(shape_name)
        data = fn(**params_dict)
        if isinstance(data, Geometry):
            return data
        return Geometry.from_lines(data)

    # 量子化ユーティリティは common.param_utils.signature_tuple を使用

    # 形状生成ヘルパは `_generate_shape_resolved` に集約

    # _params_to_tuple は廃止（署名は params_signature で統一）

    def _build_shape_method(self, name: str) -> Callable[..., LazyGeometry]:
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

        def _shape_method(**params: Any) -> LazyGeometry:
            if not is_shape_registered(name):
                # 登録解除と整合を取るため、キャッシュ済みの属性を破棄して AttributeError を送出
                self.__dict__.pop(name, None)
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
            runtime = get_active_runtime()
            fn = get_shape_generator(name)
            if runtime is not None:
                # ランタイムで最終値へ解決（量子化を含む）
                resolved = dict(runtime.before_shape_call(name, fn, dict(params)))
                params_tuple = _params_signature(fn, resolved)
                _record_spec(name, params_tuple)
                return self._lazy_shape(name, params_tuple)
            params_tuple = _params_signature(fn, dict(params))
            _record_spec(name, params_tuple)
            return self._lazy_shape(name, params_tuple)

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

    def __getattr__(self, name: str) -> Callable[..., LazyGeometry]:
        """レジストリに基づき `G.<name>` を遅延生成する。

        Parameters
        ----------
        name : str
            形状名（レジストリ登録済みキー）。

        Returns
        -------
        Callable[..., LazyGeometry]
            `G.<name>(**params) -> LazyGeometry` を返す呼び出し可能。

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
        """軽量 spec キャッシュ（統計のみ）をクリア。"""
        global _spec_seen, _spec_hits, _spec_misses
        _spec_seen.clear()
        _spec_hits = 0
        _spec_misses = 0

    @classmethod
    def cache_info(cls) -> dict[str, int]:
        """軽量 spec キャッシュの統計情報を返す。"""
        return {
            "hits": _spec_hits,
            "misses": _spec_misses,
            "maxsize": _SPEC_CACHE_MAXSIZE,
            "size": len(_spec_seen),
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
