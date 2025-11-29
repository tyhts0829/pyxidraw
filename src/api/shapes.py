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
- 形状結果の LRU は `engine.core.lazy_geometry` 側にあり、環境変数
  `PXD_SHAPE_CACHE_MAXSIZE` で制御する。各シェイプは純粋関数（副作用なし）を前提。

Design
------
- 単一入口: 形状生成は `G` が一手に引き受ける。
- 責務境界: 生成は `G`/各 shape 関数、変換は `Geometry`、加工は `E.pipeline`。
 - キャッシュ: 形状生成結果の LRU は `engine.core.lazy_geometry` に集約（`PXD_SHAPE_CACHE_MAXSIZE` で制御）。
- 動的ディスパッチ: インスタンス `__getattr__` で遅延解決し、`G.sphere(...)` の形で提供。
- 再現性と性能: `params_signature()` でパラメータを量子化→ハッシュ可能に正規化し、
  プロセス内 LRU を適用。
- 例外方針: 未登録名は `AttributeError`。生成器側の失敗は各シェイプが責任。

Examples
--------
    from api import G, E

    g = (G.sphere(subdivisions=1)
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
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

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


@dataclass
class _ShapeCallContext:
    """単一の shape 呼び出しコンテキスト（label 付き）。"""

    label: Optional[str] = None

    def with_label(self, label: str | None) -> "_ShapeCallContext":
        try:
            text = str(label or "").strip()
        except Exception:
            text = ""
        return _ShapeCallContext(label=text or None)


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
        g1 = G.sphere(subdivisions=1)
        g2 = G.polygon(n_sides=6)
    """

    def __init__(self, *, _context: _ShapeCallContext | None = None) -> None:
        # label などの補助情報を保持する呼び出しコンテキスト
        self._context = _context or _ShapeCallContext()

    @staticmethod
    def _lazy_shape(shape_name: str, params_dict: dict[str, Any]) -> LazyGeometry:
        """登録シェイプを spec（非量子化パラメータ）として Lazy に構築する。

        base_payload には shape 名も含めて格納し、後段（Parameter GUI など）で
        ラベル付けや識別に利用できるようにする。
        """
        fn = get_shape_generator(shape_name)
        impl = getattr(fn, "__shape_impl__", fn)
        # base_payload: (shape_name, shape_impl, params_dict)
        return LazyGeometry(base_kind="shape", base_payload=(shape_name, impl, dict(params_dict)))

    # 直接実行のヘルパは不要（LazyGeometry.realize 側で統一処理）

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
        Callable[..., LazyGeometry]
            `G.<name>(**params) -> LazyGeometry` となる呼び出し可能。

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
            ctx_label: str | None = getattr(self._context, "label", None)
            fn = get_shape_generator(name)
            impl = getattr(fn, "__shape_impl__", fn)

            if runtime is not None:
                # ランタイムで最終値へ解決（量子化を含む）
                resolved = dict(runtime.before_shape_call(name, fn, dict(params)))
                # 署名は impl を対象に（鍵は量子化、実行は非量子化を payload に保持）
                params_tuple = _params_signature(impl, resolved)
                _record_spec(name, params_tuple)
                lazy = self._lazy_shape(name, resolved)
            else:
                # ランタイム無し: 実値をそのまま payload に、鍵は量子化した署名
                params_dict = dict(params)
                params_tuple = _params_signature(impl, params_dict)
                _record_spec(name, params_tuple)
                lazy = self._lazy_shape(name, params_dict)

            # shape 呼び出しに対するラベルがあれば LazyGeometry 側に委譲
            if ctx_label:
                try:
                    return lazy.label(ctx_label)
                except Exception:
                    return lazy
            return lazy

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

    # --- ラベル指定（shape ヘッダ名のベース） ---

    def label(self, uid: str) -> "ShapesAPI":
        """次の shape 呼び出しに対して表示ラベルを設定する。

        例: `G.label("title").text(...)` のように使い、Parameter GUI 上の
        shape ヘッダ名に `title`, `title_1` ... のようなラベルを用いる。
        """
        new_ctx = (self._context or _ShapeCallContext()).with_label(uid)
        return ShapesAPI(_context=new_ctx)

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
