from __future__ import annotations

"""テスト/スタブ生成で使用するダミー依存のインストーラを委譲するヘルパー。"""


def install() -> None:
    """`scripts.dummy_deps.install` を呼び出してダミー依存を挿入する。

    - `numba.njit` の no-op デコレータ
    - `fontTools` の最小スタブ
    - `shapely` の最小ジオメトリクラス
    """
    # import を関数内に置き、import 時の副作用を避ける
    from scripts.dummy_deps import install as _install  # type: ignore

    _install()
