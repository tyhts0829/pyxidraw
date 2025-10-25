"""
どこで: `api.effects`（エフェクト・パイプラインの高レベル API）。
何を: 登録エフェクトの直列適用を宣言・実行する Pipeline を提供（外部保存/復元/仕様検証の API は提供しない）。
なぜ: Geometry を入力に副作用なく決定的に加工し、ユーザーが build/キャッシュを意識せずに再計算を最小化するため。

提供コンポーネント:
- `Pipeline`: 宣言（ステップ列）と実行を担う単一オブジェクト。`__call__(g)` で逐次適用。
  - 実行時にパラメータを解決・量子化し、差分があれば裏で再ビルド。
  - 単層 LRU 風キャッシュを内蔵。鍵は `(geometry_digest, pipeline_key)`。
  - `clear_cache()` でキャッシュを手動クリア可能。
- `PipelineBuilder`: 互換のための薄いビルダー。`.build()` は no-op 同義で `Pipeline` を返す。
- `PipelineStep`: 1 ステップの定義（`name` と `params`）。
- `E`: 利用者向けシングルトン（`from api import E`）。`E.pipeline` が `PipelineBuilder` を返す。

キャッシュ設計:
- `Geometry.digest`（16B の blake2b）と、解決後（量子化後）パラメータから導出した `pipeline_key` を組み合わせる。
- 容量は `.cache(maxsize=...)` または環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` で制御。
  - `None`: 無制限、`0`: 無効、正の整数で上限。
- 実装は `OrderedDict` による単純な LRU 風。ヒット時は末尾に移動し、上限超過で先頭から追い出し。

ハッシュと同一性:
- `pipeline_key` は各ステップの「名前」「関数バイトコード近似」「量子化後パラメータの決定的表現」から blake2b を積み上げて算出し、16B にまとめる。
- `Geometry` との組み合わせにより処理結果のキャッシュ同一性が安定する。

スレッド/プロセスについての注意:
- キャッシュは `Pipeline` インスタンスに閉じた `OrderedDict`。共有時は外部でロックを用いるか、スレッドごとに別インスタンスを使用する。
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Sequence

import numpy as np

from common.param_utils import params_signature as _params_signature
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


def _params_digest_from_tuple(params_tuple: tuple[tuple[str, object], ...]) -> bytes:
    """正規化タプルから短いハッシュへ圧縮。"""
    data = repr(params_tuple).encode()
    return hashlib.blake2b(data, digest_size=8).digest()


@dataclass(frozen=True)
class PipelineStep:
    name: str
    params: dict[str, Any]


# ---- Global compiled cache (reuse across Builders/Pipelines) --------------
_GLOBAL_COMPILED_CACHE: "OrderedDict[tuple[tuple[str, tuple[tuple[str, object], ...]], ...], _CompiledPipeline]" = OrderedDict()  # type: ignore[name-defined]
_GLOBAL_LOCK = RLock()
_GLOBAL_MAXSIZE_ENV = os.getenv("PXD_COMPILED_CACHE_MAXSIZE")
try:
    _GLOBAL_MAXSIZE: int | None = (
        int(_GLOBAL_MAXSIZE_ENV) if _GLOBAL_MAXSIZE_ENV is not None else None
    )
    if _GLOBAL_MAXSIZE is not None and _GLOBAL_MAXSIZE < 0:
        _GLOBAL_MAXSIZE = 0
except Exception:
    _GLOBAL_MAXSIZE = None


class _CompiledPipeline:
    """量子化後パラメータで固定化された実行体（インスタンスローカル LRU を保持）。"""

    def __init__(
        self,
        steps: Sequence[tuple[str, Callable[..., Geometry], tuple[tuple[str, object], ...]]],
        *,
        cache_maxsize: int | None,
    ) -> None:
        self._steps = list(steps)
        self._cache_maxsize = cache_maxsize
        self._cache: "OrderedDict[tuple[bytes, bytes], Geometry]" = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evicts = 0

        # pipeline_key（16B）を計算
        h = hashlib.blake2b(digest_size=16)
        for name, fn, params_tuple in self._steps:
            h.update(name.encode())
            h.update(_fn_version(fn))
            h.update(_params_digest_from_tuple(params_tuple))
        self._pipeline_key = h.digest()

    def __call__(self, g: Geometry) -> Geometry:
        key = (_geometry_hash(g), self._pipeline_key)
        with self._lock:
            if key in self._cache:
                out = self._cache.pop(key)
                self._cache[key] = out
                self._hits += 1
                return out

        out = g
        for name, fn, params_tuple in self._steps:
            params = dict(params_tuple)
            out = fn(out, **params)

        if self._cache_maxsize == 0:
            return out
        with self._lock:
            self._cache[key] = out
            if self._cache_maxsize is not None and self._cache_maxsize > 0:
                while len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)
                    self._evicts += 1
            self._misses += 1
        return out

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evicts = 0

    def cache_info(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._cache_maxsize if self._cache_maxsize is not None else -1,
                "hits": self._hits,
                "misses": self._misses,
                "evicts": self._evicts,
            }


def global_cache_counters() -> dict[str, int]:
    """エフェクト（compiled pipelines）全体のヒット/ミス累計を集計する。

    Returns
    -------
    dict[str, int]
        以下のキーを持つ辞書。
        - "compiled": 現在存在する compiled pipelines の数
        - "enabled": そのうちキャッシュが有効（maxsize!=0）なものの数
        - "hits": 全 compiled の `_hits` 合計
        - "misses": 全 compiled の `_misses` 合計

    Notes
    -----
    UI/HUD 側で前後差分から HIT/MISS を判定するのに用いる。
    """
    with _GLOBAL_LOCK:
        compiled_list = list(_GLOBAL_COMPILED_CACHE.values())
        compiled = len(compiled_list)
        enabled = 0
        hits = 0
        misses = 0
        for cp in compiled_list:
            try:
                # 有効判定
                if getattr(cp, "_cache_maxsize", None) != 0:
                    enabled += 1
                hits += int(getattr(cp, "_hits", 0))
                misses += int(getattr(cp, "_misses", 0))
            except Exception:
                # 不正アクセスは無視（集計に影響しない）
                pass
        return {"compiled": compiled, "enabled": enabled, "hits": hits, "misses": misses}


class Pipeline:
    def __init__(
        self,
        steps: Sequence[PipelineStep],
        *,
        cache_maxsize: int | None = 128,
        pipeline_uid: str = "",
    ):
        self._steps = list(steps)
        self._cache_maxsize: int | None = cache_maxsize
        self._compiled: _CompiledPipeline | None = None
        self._compiled_signature: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] | None = (
            None
        )
        self._lock = RLock()
        # Parameter GUI の一意識別に用いるラベル（空文字なら従来表記）
        self._pipeline_uid: str = str(pipeline_uid or "")

    def _ensure_compiled(
        self, runtime_signature: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    ) -> _CompiledPipeline:
        # まずグローバルキャッシュで再利用を試みる
        with _GLOBAL_LOCK:
            if runtime_signature in _GLOBAL_COMPILED_CACHE:
                compiled = _GLOBAL_COMPILED_CACHE.pop(runtime_signature)
                _GLOBAL_COMPILED_CACHE[runtime_signature] = compiled
                # Builder の cache_maxsize 設定を反映（再利用時にも尊重）
                try:
                    compiled._cache_maxsize = self._cache_maxsize  # type: ignore[attr-defined]
                    if self._cache_maxsize == 0:
                        compiled.clear_cache()
                except Exception:
                    pass
            else:
                compiled_steps: list[
                    tuple[str, Callable[..., Geometry], tuple[tuple[str, object], ...]]
                ] = []
                for name, params_tuple in runtime_signature:
                    fn = get_effect(name)
                    compiled_steps.append((name, fn, params_tuple))
                compiled = _CompiledPipeline(compiled_steps, cache_maxsize=self._cache_maxsize)
                _GLOBAL_COMPILED_CACHE[runtime_signature] = compiled
                if _GLOBAL_MAXSIZE is not None and _GLOBAL_MAXSIZE > 0:
                    while len(_GLOBAL_COMPILED_CACHE) > _GLOBAL_MAXSIZE:
                        _GLOBAL_COMPILED_CACHE.popitem(last=False)
        # ローカルにも保持（軽量参照）
        with self._lock:
            self._compiled = compiled
            self._compiled_signature = runtime_signature
            return compiled

    def __call__(self, g: Geometry) -> Geometry:
        # 各ステップの params を Runtime で解決→量子化→署名タプル列へ
        runtime = get_active_runtime()
        # パイプライン UID: 明示 label を優先、未設定ならランタイムの逐次 UID（フレーム内安定）を使用
        pipeline_uid = self._pipeline_uid
        if not pipeline_uid and runtime is not None:
            try:
                # ParameterRuntime/SnapshotRuntime の両方で next_pipeline_uid を提供（無ければ空）
                pipeline_uid = str(getattr(runtime, "next_pipeline_uid")())  # type: ignore[misc]
            except Exception:
                pipeline_uid = ""
        runtime_sig_list: list[tuple[str, tuple[tuple[str, object], ...]]] = []
        for idx, st in enumerate(self._steps):
            fn = get_effect(st.name)
            params = dict(st.params)
            if runtime is not None:
                params = dict(
                    runtime.before_effect_call(
                        step_index=idx,
                        effect_name=st.name,
                        fn=fn,
                        params=params,
                        pipeline_uid=pipeline_uid,
                    )
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
            params_tuple = _params_signature(fn, params)
            runtime_sig_list.append((st.name, params_tuple))

        runtime_signature = tuple(runtime_sig_list)
        compiled = self._ensure_compiled(runtime_signature)
        return compiled(g)

    def clear_cache(self) -> None:
        with self._lock:
            if self._compiled is not None:
                self._compiled.clear_cache()

    def cache_info(self) -> dict[str, int]:
        with self._lock:
            if self._compiled is not None:
                return self._compiled.cache_info()
            return {"size": 0, "maxsize": -1, "hits": 0, "misses": 0, "evicts": 0}

    def __repr__(self) -> str:  # 開発時の可読性向上
        steps = ", ".join(
            f"{s.name}({', '.join(f'{k}={v!r}' for k, v in s.params.items())})" for s in self._steps
        )
        if self._pipeline_uid:
            return f"Pipeline(uid={self._pipeline_uid!r}, steps=[{steps}], cache_maxsize={self._cache_maxsize})"
        return f"Pipeline(steps=[{steps}], cache_maxsize={self._cache_maxsize})"

    __str__ = __repr__


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
        self._pipeline: Pipeline | None = None
        self._uid: str | None = None
        _env = os.getenv("PXD_PIPELINE_CACHE_MAXSIZE")
        if _env is not None:
            try:
                val = int(_env)
                self._cache_maxsize = 0 if val < 0 else val
            except ValueError:
                pass

    def _add(self, name: str, params: dict[str, Any]) -> "PipelineBuilder":
        self._steps.append(PipelineStep(name, params))
        self._pipeline = None  # invalidate
        return self

    def __getattr__(self, name: str):
        # 動的にエフェクト名を受け取り、paramsを蓄積
        def adder(**params):
            return self._add(name, params)

        return adder

    # オプション: 単層キャッシュの上限を設定（None で無制限）
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
        self._pipeline = None  # invalidate
        return self

    # 任意: パイプラインの UI 識別子を明示設定（未設定時は自動採番）
    def label(self, uid: str) -> "PipelineBuilder":
        self._uid = str(uid)
        self._pipeline = None  # invalidate
        return self

    def _ensure_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            uid = self._uid if self._uid is not None else ""
            self._pipeline = Pipeline(
                self._steps, cache_maxsize=self._cache_maxsize, pipeline_uid=uid
            )
        return self._pipeline

    def build(self) -> Pipeline:
        # 実体を生成して保持（以後の __call__/cache_info で再利用）。
        # 厳格検証は行わない（未知キーはランタイム呼び出し時のシグネチャで検出され得る）。
        return self._ensure_pipeline()

    def __call__(self, g: Geometry) -> Geometry:
        return self._ensure_pipeline()(g)

    # 便宜: Builder にも統計/クリアを露出（利用者が build を意識しなくてよい）
    def cache_info(self) -> dict[str, int]:
        return self._ensure_pipeline().cache_info()

    def clear_cache(self) -> None:
        self._ensure_pipeline().clear_cache()


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
# to_spec/from_spec は削除（API 縮減）


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


# 仕様検証 API は提供しない（縮減方針）。
_PIPELINE_UID_COUNTER = 0
_PIPELINE_UID_LOCK = RLock()


def _next_pipeline_uid() -> str:
    global _PIPELINE_UID_COUNTER
    with _PIPELINE_UID_LOCK:
        uid = f"p{_PIPELINE_UID_COUNTER}"
        _PIPELINE_UID_COUNTER += 1
        return uid
