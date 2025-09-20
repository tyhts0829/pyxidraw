"""
api.effects — パイプライン実行モジュール（Effects オーケストレーター）

本モジュールは、登録済みエフェクト群（`effects.registry`）を直列に適用する
「パイプライン」を提供する。入力は唯一の幾何表現 `Geometry`（engine.core.geometry）で、
各ステップは `Geometry -> Geometry` の純関数として実行される。副作用を避け、
決定的で再現性の高い処理チェーンを組み立てることを目的とする。

提供コンポーネント:
- `Pipeline`: 不変な処理定義を保持し、`__call__(g)` で逐次適用する実行体。
  - 単層 LRU 風キャッシュを内蔵。鍵は `(geometry_digest, pipeline_key)`。
  - `clear_cache()` でキャッシュを手動クリア可能。
  - `to_spec()/from_spec()` により JSON 風の仕様へシリアライズ/デシリアライズ可能。
- `PipelineBuilder`: ビルダー（フルエント API）。`E.pipeline ... .build()` で `Pipeline` を生成。
  - 動的属性でエフェクト名を受け取り、`adder(**params)` でステップを追加する仕組み。
  - `.cache(maxsize=...)` で単層キャッシュの上限を設定（`None`=無制限、`0`=無効）。
  - `.strict(enabled=True)` でビルド時の厳格検証を有効化（未知パラメータを検出して `TypeError`）。
- `PipelineStep`: 1 ステップの定義（`name` と `params`）。
- `E`: 利用者向けシングルトン（`from api import E`）。`E.pipeline` が `PipelineBuilder` を返す。
- `validate_spec(spec)`: 仕様配列を検証するユーティリティ（不正時に `TypeError/KeyError`）。

キャッシュ設計:
- `Geometry.digest`（16B の blake2b 指紋）と、ステップ列から導出した `pipeline_key` を組み合わせる。
- `Geometry.digest` が無効/未計算の場合でも、配列内容から都度ハッシュを計算してフォールバック。
- 容量は `PipelineBuilder.cache(maxsize=...)` または環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` で制御。
  - `None`: 無制限（従来互換）。`0`: キャッシュ無効。
- 実装は `OrderedDict` による単純な LRU 風。ヒット時は末尾に移動し、上限超過で先頭から追い出し。

厳格検証（strict）:
- 各ステップの `params` キーを該当エフェクト関数のシグネチャと照合。
- `g` 引数は内部で供給するため指定不要。`**kwargs` を受け取る関数は未知キー許容（Builder 側）。
- `validate_spec` はデータ受け渡し境界（設定/ファイル）での安全性確保に重点を置き、
  JSON 風の値かどうかのヒューリスティックも併せて確認する。

ハッシュと同一性:
- `pipeline_key` は各ステップの「名前」「関数バイトコード（近似版）」「パラメータ正規化結果」
  から blake2b（8B）を積み上げて算出し、最終的に 16B にまとめた指紋を使用する。
- `Geometry` との組み合わせにより処理結果のキャッシュ同一性が安定し、
  大規模な入出力でも再計算を抑制できる。

スレッド/プロセスについての注意:
- 現実装のキャッシュはモジュール内 `Pipeline` インスタンスに閉じた `OrderedDict` を使用。
  マルチスレッドで同一インスタンスを共有する場合は外部でロックを用いるか、
  スレッドごとに別インスタンスを使用することを推奨。

使用例:
    from api import E, G
    g = G.grid(subdivisions=(0.5, 0.5))
    pipe = (
        E.pipeline
         .displace(amplitude_mm=0.05)
         .rotate(angles_rad=(0.0, 0.0, 0.3))
         .cache(maxsize=128)
         .strict(True)
         .build()
    )
    out = pipe(g)  # Geometry -> Geometry

モジュール境界:
- エフェクト解決は `effects.registry.get_effect` に委譲（未登録名は `KeyError`）。
- ジオメトリ表現は `engine.core.geometry.Geometry` に統一。
- 本モジュールは I/O を持たず、エフェクト適用の順序・同一性・キャッシュ管理に専念する。
"""

from __future__ import annotations

import hashlib
import inspect
import os
from collections import OrderedDict
from dataclasses import dataclass
from itertools import zip_longest
from threading import RLock
from typing import Any, Callable, Sequence

import numpy as np

from common.param_utils import params_to_tuple as _params_to_tuple
from effects.registry import get_effect
from engine.core.geometry import Geometry
from engine.ui.parameters import get_active_runtime
from engine.ui.parameters.runtime import resolve_without_runtime


def _geometry_hash(g: Geometry) -> bytes:
    """Geometry のハッシュ。

    Geometry が `digest` プロパティを提供する場合はそれを優先し、
    なければ従来通りに coords/offsets から計算します。
    """
    try:
        return g.digest  # type: ignore[attr-defined]
    except Exception:
        # フォールバック（理論上到達しない想定）
        c, o = g.as_arrays(copy=False)
        c = np.ascontiguousarray(c).view(np.uint8)
        o = np.ascontiguousarray(o).view(np.uint8)
        h = hashlib.blake2b(digest_size=16)
        # Fallback 経路のためコピーを許容し、bytes に変換して渡す
        h.update(c.tobytes())
        h.update(o.tobytes())
        return h.digest()


def _fn_version(fn: Callable[..., Geometry]) -> bytes:
    code = getattr(fn, "__code__", None)
    data = code.co_code if code else repr(fn).encode()
    return hashlib.blake2b(data, digest_size=8).digest()


def _params_digest(params: dict[str, Any]) -> bytes:
    """パラメータを共通実装で正規化し、短いハッシュへ圧縮。"""
    normalized = _params_to_tuple(params)
    data = repr(normalized).encode()
    return hashlib.blake2b(data, digest_size=8).digest()


@dataclass(frozen=True)
class PipelineStep:
    name: str
    params: dict[str, Any]


class Pipeline:
    def __init__(self, steps: Sequence[PipelineStep], *, cache_maxsize: int | None = 128):
        self._steps = list(steps)
        # LRU 互換の単層キャッシュ（maxsize=None なら従来通り無制限）
        self._cache_maxsize: int | None = cache_maxsize
        self._cache: "OrderedDict[tuple[bytes, bytes], Geometry]" = OrderedDict()
        self._lock = RLock()

        # パイプラインハッシュを先に計算
        h = hashlib.blake2b(digest_size=16)
        for st in self._steps:
            fn = get_effect(st.name)
            h.update(st.name.encode())
            h.update(_fn_version(fn))
            h.update(_params_digest(st.params))
        self._pipeline_key = h.digest()

    def __call__(self, g: Geometry) -> Geometry:
        key = (_geometry_hash(g), self._pipeline_key)
        with self._lock:
            if key in self._cache:
                # LRU: 参照を末尾へ
                out = self._cache.pop(key)
                self._cache[key] = out
                return out

        runtime = get_active_runtime()
        out = g
        for idx, st in enumerate(self._steps):
            fn = get_effect(st.name)
            params = dict(st.params)
            if runtime is not None:
                params = runtime.before_effect_call(
                    step_index=idx,
                    effect_name=st.name,
                    fn=fn,
                    params=params,
                )
            else:
                params = dict(
                    resolve_without_runtime(
                        scope="effect",
                        name=st.name,
                        fn=fn,
                        params=params,
                        index=idx,
                    )
                )
            out = fn(out, **params)

        # 単層キャッシュ（LRU 風）
        if self._cache_maxsize == 0:
            return out  # キャッシュ無効
        with self._lock:
            self._cache[key] = out
            if self._cache_maxsize is not None and self._cache_maxsize > 0:
                while len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)  # 先頭（最古）を追い出す
        return out

    def clear_cache(self) -> None:
        """パイプラインの単層キャッシュをクリア。"""
        with self._lock:
            self._cache.clear()

    def cache_info(self) -> dict[str, int]:
        """簡易キャッシュ情報（サイズのみ）。"""
        with self._lock:
            return {"size": len(self._cache)}

    def __repr__(self) -> str:  # 開発時の可読性向上
        steps = ", ".join(
            f"{s.name}({', '.join(f'{k}={v!r}' for k, v in s.params.items())})" for s in self._steps
        )
        return f"Pipeline(steps=[{steps}], cache_maxsize={self._cache_maxsize})"

    __str__ = __repr__

    # ---- Serialization (Proposal 6) ----
    def to_spec(self) -> list[dict[str, Any]]:
        """シリアライズ可能な仕様を返す: `[{"name": str, "params": dict}]`。"""
        return [{"name": s.name, "params": dict(s.params)} for s in self._steps]

    @staticmethod
    def from_spec(spec: Sequence[dict[str, Any]]) -> "Pipeline":
        """仕様から `Pipeline` を生成。不正な形状/エフェクト名の場合は例外を送出。"""
        validate_spec(spec)
        steps: list[PipelineStep] = [
            PipelineStep(str(entry["name"]), dict(entry.get("params", {}))) for entry in spec  # type: ignore[arg-type]
        ]
        return Pipeline(steps)


class PipelineBuilder:
    """エフェクトパイプラインを段階的に構築するビルダー。

    Notes
    -----
    UI/ランタイム層経由の利用では、利用者が 0..1 の正規化値を指定すると
    `engine.ui.parameters` が実レンジへ変換してから各エフェクトへ引き渡す。
    本ビルダーを直接呼び出す場合は、各エフェクト関数が期待する実レンジ
    （例: ミリメートル、ラジアン）でパラメータを渡すこと。
    """

    def __init__(self):
        self._steps: list[PipelineStep] = []
        # 既定サイズは環境変数から上書き可能
        self._cache_maxsize: int | None = None
        # クリーン化方針: 既定で厳格検証を有効化
        self._strict: bool = True
        _env = os.getenv("PXD_PIPELINE_CACHE_MAXSIZE")
        if _env is not None:
            try:
                val = int(_env)
                self._cache_maxsize = 0 if val < 0 else val
            except ValueError:
                pass

    def _add(self, name: str, params: dict[str, Any]) -> "PipelineBuilder":
        self._steps.append(PipelineStep(name, params))
        return self

    def __getattr__(self, name: str):
        # 動的にエフェクト名を受け取り、paramsを蓄積
        def adder(**params):
            return self._add(name, params)

        return adder

    # オプション: 単層キャッシュの上限を設定（None で無制限・従来互換）
    def cache(self, *, maxsize: int | None) -> "PipelineBuilder":
        """パイプライン結果の単層キャッシュ上限を設定する。

        Parameters
        ----------
        maxsize : int or None
            キャッシュに保持する最大エントリ数。``None`` で無制限、``0`` で
            キャッシュ無効、正の整数で上限を設定する。

        Notes
        -----
        既定値は ``None`` で無制限運用となる。重いジオメトリを多数扱う際は
        明示的に小さい値へ制限し、メモリ使用量の高止まりを避けることを推奨。
        """
        if maxsize is not None and maxsize < 0:
            self._cache_maxsize = 0
        else:
            self._cache_maxsize = maxsize
        return self

    # オプション: 厳格検証を有効化（ビルド時にパラメータ名を検査）
    def strict(self, enabled: bool = True) -> "PipelineBuilder":
        """ビルド時にステップのパラメータ名を検証する。

        Parameters
        ----------
        enabled : bool, default True
            ``True`` で厳格検証を有効化。``False`` にすると未知パラメータを
            許容し、開発初期の試行錯誤を優先できる。

        Notes
        -----
        厳格検証が有効な場合、各ステップのパラメータ名をエフェクト関数の
        シグネチャと突き合わせ、未知キーがあれば ``TypeError`` を送出する。
        ``**kwargs`` を受け取る関数は未知キーを許容する。
        """
        self._strict = enabled
        return self

    def build(self) -> Pipeline:
        if self._strict:
            for i, st in enumerate(self._steps):
                fn = get_effect(st.name)
                try:
                    sig = inspect.signature(fn)
                except ValueError:
                    # Builtins などはスキップ
                    continue
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if has_var_kw:
                    continue
                allowed = {
                    p.name
                    for p in sig.parameters.values()
                    if p.kind
                    in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                }
                if "g" in allowed:
                    allowed.remove("g")
                unknown = [k for k in st.params.keys() if k not in allowed]
                if unknown:
                    allowed_sorted = ", ".join(sorted(allowed))
                    raise TypeError(
                        f"step[{i}] effect '{st.name}' has unknown params: {unknown}. Allowed: [{allowed_sorted}]"
                    )
        return Pipeline(self._steps, cache_maxsize=self._cache_maxsize)

    def __call__(self, g: Geometry) -> Geometry:
        return self.build()(g)


class EffectsAPI:
    @property
    def pipeline(self) -> PipelineBuilder:
        """エフェクトパイプラインを組み立てるビルダーを返す。

        Returns
        -------
        PipelineBuilder
            エフェクト適用ステップを連結し `Pipeline` を生成するビルダー。

        Notes
        -----
        GUI やランタイム経由で利用する場合、利用者が指定する 0..1 の正規化
        パラメータは内部で実レンジへ変換されてからビルダーに渡される。
        直接利用する場合は各エフェクトが期待する単位系で値を渡すこと。
        """
        return PipelineBuilder()


# シングルトンインスタンス（`from api import E`）
E = EffectsAPI()


# Helper functions (optional API)
def to_spec(pipeline: Pipeline) -> list[dict[str, Any]]:
    return pipeline.to_spec()


def from_spec(spec: Sequence[dict[str, Any]]) -> Pipeline:
    return Pipeline.from_spec(spec)


# ---- Spec validation (Proposal 6) -----------------------------------------
def _is_json_like(value: Any) -> bool:
    """値が JSON 風（シリアライズ可能）かをヒューリスティックに検査。"""
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_like(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_like(v) for k, v in value.items())
    return False


def validate_spec(spec: Sequence[dict[str, Any]]) -> None:
    """パイプライン仕様を検証（不正時は TypeError/KeyError）。

    仕様:
    - `spec` は `{"name": str, "params": dict}` の列（list/tuple）
    - `name` は登録済みエフェクト名
    - `params` は辞書かつ JSON 風（数値/文字列/真偽/None、入れ子の list/dict を許容）
    - 可能なら関数シグネチャと照合し、未知パラメータを検出（ただし関数が **kwargs を取る場合は許容）
    """
    if not isinstance(spec, (list, tuple)):
        raise TypeError("spec はステップ辞書の list または tuple である必要があります")

    for i, entry in enumerate(spec):
        if not isinstance(entry, dict):
            raise TypeError(
                f"spec[{i}] は dict である必要があります（実際: {type(entry).__name__}）"
            )
        name = entry.get("name")
        params = entry.get("params", {})
        if not isinstance(name, str):
            raise TypeError(f"spec[{i}]['name'] は str である必要があります")
        if not isinstance(params, dict):
            raise TypeError(f"spec[{i}]['params'] は dict である必要があります")

        # Validate effect registration
        fn = get_effect(name)  # raises KeyError if not registered

        # Validate params JSON-likeness
        for k, v in params.items():
            if not isinstance(k, str):
                raise TypeError(f"spec[{i}]['params'] のキーは str である必要があります: {k!r}")
            if not _is_json_like(v):
                raise TypeError(
                    f"spec[{i}]['params']['{k}'] は JSON 風の値ではありません: {type(v).__name__}"
                )

        # 厳格: 未知パラメータを禁止（**kwargs 許容の特例も廃止）
        try:
            sig = inspect.signature(fn)
        except ValueError:
            # 署名を取得できない場合はスキップ（ほぼ発生しない想定）
            continue
        allowed = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        # 効果関数の第1引数 g は外部指定しない
        if "g" in allowed:
            allowed.remove("g")
        unknown = [k for k in params.keys() if k not in allowed]
        if unknown:
            allowed_sorted = ", ".join(sorted(allowed))
            raise TypeError(
                f"spec[{i}] effect '{name}' has unknown params: {unknown}. Allowed: [{allowed_sorted}]"
            )

        # （重複させない）この時点で未知パラメータは検出済み

        # Optional: param meta validation (type/range/choices) if effect exposes __param_meta__
        meta = getattr(fn, "__param_meta__", None)
        if isinstance(meta, dict):
            for k, rules in meta.items():
                if k not in params:
                    continue
                v = params[k]
                # type check (loose)
                t = rules.get("type") if isinstance(rules, dict) else None
                if t == "number" and not isinstance(v, (int, float)):
                    raise TypeError(
                        f"spec[{i}]['params']['{k}'] は数値である必要があります（実際: {type(v).__name__}）"
                    )
                if t == "integer" and not isinstance(v, int):
                    raise TypeError(
                        f"spec[{i}]['params']['{k}'] は整数である必要があります（実際: {type(v).__name__}）"
                    )
                if t == "string" and not isinstance(v, str):
                    raise TypeError(
                        f"spec[{i}]['params']['{k}'] は文字列である必要があります（実際: {type(v).__name__}）"
                    )
                if t == "vec3":
                    # allow scalar, 1-tuple, or 3-tuple of numbers
                    def _is_num(x):
                        return isinstance(x, (int, float))

                    if _is_num(v):
                        pass
                    elif (
                        isinstance(v, (list, tuple))
                        and len(v) in (1, 3)
                        and all(_is_num(x) for x in v)
                    ):
                        pass
                    else:
                        raise TypeError(
                            f"spec[{i}]['params']['{k}'] は数値のスカラー、1要素、または3要素のタプルである必要があります"
                        )
                # range
                if isinstance(rules, dict):
                    min_rule = rules.get("min")
                    max_rule = rules.get("max")

                    def _iter_components(val: Any) -> list[float]:
                        if isinstance(val, (list, tuple)):
                            return [float(x) for x in val if isinstance(x, (int, float))]
                        if isinstance(val, (int, float)):
                            return [float(val)]
                        return []

                    values = _iter_components(v)

                    if min_rule is not None and values:
                        mins = _iter_components(min_rule) or [float(min_rule)]
                        for idx, (val, min_val) in enumerate(
                            zip_longest(values, mins, fillvalue=mins[-1])
                        ):
                            if val < min_val:
                                raise TypeError(
                                    f"spec[{i}]['params']['{k}'][{idx}]={val} は最小値 {min_rule} 未満です"
                                )
                    if max_rule is not None and values:
                        maxs = _iter_components(max_rule) or [float(max_rule)]
                        for idx, (val, max_val) in enumerate(
                            zip_longest(values, maxs, fillvalue=maxs[-1])
                        ):
                            if val > max_val:
                                raise TypeError(
                                    f"spec[{i}]['params']['{k}'][{idx}]={val} は最大値 {max_rule} を超えています"
                                )
                    # choices
                    choices = rules.get("choices")
                    if choices is not None and v not in choices:
                        raise TypeError(
                            f"spec[{i}]['params']['{k}']={v!r} は {choices} のいずれかである必要があります"
                        )
