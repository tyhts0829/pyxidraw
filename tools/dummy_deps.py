from __future__ import annotations

"""
テスト/ツール向けの軽量依存を補うダミーモジュールインストーラ

目的:
- 開発・CI・スタブ生成などの軽量環境で、重量級の任意依存関係（numba / fontTools / shapely）が
  未インストールでもスクリプトやテストが実行できるよう、最小限の互換シムを `sys.modules` に
  注入する。

提供するシムの概要:
- `numba.njit`: 何もしないデコレータ（関数をそのまま返す）。
- `fontTools` の一部: `pens.recordingPen.RecordingPen` と `ttLib.TTFont` を最小実装。
  - `TTFont.__getitem__("head")["unitsPerEm"]` 相当の参照のみが成立するよう、
    `unitsPerEm=1000` を返す簡易オブジェクトを用意。
- `shapely` の一部: `geometry` と `geometry.base` を最小実装し、
  `LineString`, `MultiLineString`, `Polygon`, `MultiPolygon`, `BaseGeometry` を提供。
  - 幾何クラスは空ジオメトリ相当のプロパティ（`is_empty=True`, `geoms=[]`, `coords=[]`）を持ち、
    `buffer(...)` は自己を返すだけとする。

注意/制約:
- 本シムは本プロジェクトのテストとスタブ生成が通るための最小形であり、完全互換ではない。
- 実環境・本番用途では実パッケージの導入を推奨。
- いずれも `# pragma: no cover` を付け、分岐網羅をテスト対象から除外している。

使い方:
    from tools.dummy_deps import install
    install()  # 必要に応じて早期に呼び出し、欠落依存を埋める

関連:
- `shapes/text` は fontTools、`effects/offset` は shapely を参照する。
  それらが未インストールでも最低限の経路が動作するよう設計している。
"""

import sys
import types


def install() -> None:
    """欠落している任意依存（numba/fontTools/shapely）に対し、最小限の代替モジュールを注入する。"""

    # numba のシム注入（存在すれば何もしない）
    try:  # pragma: no cover - import path（実パッケージが入っている場合）
        import numba  # noqa: F401
    except Exception:  # pragma: no cover - shim path（未導入の場合のフォールバック）
        m = types.ModuleType("numba")

        def _njit(*_a, **_k):
            """何もしないデコレータ。関数をそのまま返す。"""

            def deco(fn):
                return fn

            return deco

        m.njit = _njit  # type: ignore[attr-defined]
        sys.modules["numba"] = m

    # fontTools のシム注入（pens/ttLib の必要最小のみ）
    try:  # pragma: no cover
        import fontTools  # noqa: F401
    except Exception:  # pragma: no cover
        ft = types.ModuleType("fontTools")
        pens = types.ModuleType("fontTools.pens")
        rec = types.ModuleType("fontTools.pens.recordingPen")

        class RecordingPen:  # pragma: no cover - dummy
            """最小の RecordingPen。呼び出しは無視する。"""

            def __init__(self, *a, **k):
                pass

        rec.RecordingPen = RecordingPen  # type: ignore
        ttLib = types.ModuleType("fontTools.ttLib")

        class TTFont:  # pragma: no cover - dummy
            """最小の TTFont。`__getitem__` の一部参照のみをサポート。"""

            def __init__(self, *a, **k):
                pass

            def __getitem__(self, key):
                # `font["head"].unitsPerEm` 相当の参照を簡易に満たす
                return types.SimpleNamespace(unitsPerEm=1000)

        ttLib.TTFont = TTFont  # type: ignore
        sys.modules["fontTools"] = ft
        sys.modules["fontTools.pens"] = pens
        sys.modules["fontTools.pens.recordingPen"] = rec
        sys.modules["fontTools.ttLib"] = ttLib

    # shapely のシム注入（geometry と base の必要最小のみ）
    try:  # pragma: no cover
        import shapely  # noqa: F401
    except Exception:  # pragma: no cover
        shp = types.ModuleType("shapely")
        geom = types.ModuleType("shapely.geometry")
        geom_base = types.ModuleType("shapely.geometry.base")

        class BaseGeometry:  # pragma: no cover - dummy
            """shapely の基底クラスの最小代替。"""

            pass

        class _G(BaseGeometry):  # pragma: no cover - dummy
            """空ジオメトリ相当の挙動を持つ簡易クラス。"""

            def __init__(self, *a, **k):
                self._coords = []

            @property
            def is_empty(self):
                return True

            @property
            def geoms(self):
                return []

            @property
            def coords(self):
                return self._coords

            def buffer(self, *a, **k):
                # バッファ処理は未実装。互換性のため自己を返すのみ。
                return self

        class LineString(_G):
            """最小 LineString 互換。"""

            pass

        class MultiLineString(_G):
            """最小 MultiLineString 互換。"""

            pass

        class Polygon(_G):
            """最小 Polygon 互換。`exterior.coords` のみを保持。"""

            def __init__(self, *a, **k):
                super().__init__()
                self.exterior = types.SimpleNamespace(coords=[])

        class MultiPolygon(_G):
            """最小 MultiPolygon 互換。"""

            pass

        geom_base.BaseGeometry = BaseGeometry  # type: ignore
        geom.LineString = LineString  # type: ignore
        geom.MultiLineString = MultiLineString  # type: ignore
        geom.Polygon = Polygon  # type: ignore
        geom.MultiPolygon = MultiPolygon  # type: ignore
        sys.modules["shapely"] = shp
        sys.modules["shapely.geometry"] = geom
        sys.modules["shapely.geometry.base"] = geom_base
