"""
どこで: `engine.ui.parameters` の状態管理層。
何を: ParameterDescriptor/RangeHint のメタと、ParameterStore による値（original/override/midi）を集中管理。
    購読通知と簡易レンジ推定（ParameterLayoutConfig）も提供。
なぜ: UI/ランタイムが共有する単一の真実源（Single Source of Truth）として、一貫した状態管理を担うため。

補足:
- `set_override()` は実値をそのまま保持し、クランプしない（表示上のクランプは GUI レイヤの責務）。
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Iterable, Literal, Mapping

ValueType = Literal["float", "int", "bool", "enum", "vector", "string"]
SourceType = Literal["shape", "effect"]
OverrideSource = Literal["gui"]


@dataclass(frozen=True)
class RangeHint:
    """UI 表示用の範囲ヒント（実レンジ）。"""

    min_value: float | int
    max_value: float | int
    step: float | int | None = None
    scale: str | None = None  # linear / log 等（現状は linear 固定）


@dataclass(frozen=True)
class VectorRangeHint:
    """ベクトル用の範囲ヒント（各成分ごとの実レンジ）。"""

    min_values: tuple[float, float, float] | tuple[float, float, float, float]
    max_values: tuple[float, float, float] | tuple[float, float, float, float]
    steps: (
        tuple[float | None, float | None, float | None]
        | tuple[float | None, float | None, float | None, float | None]
    )
    scale: str | None = None  # linear / log 等（現状は linear 固定）


@dataclass(frozen=True)
class ParameterDescriptor:
    """GUI に表示するパラメータのメタ情報。

    - 並び順制御のために optional メタ（pipeline_uid/step_index/param_order）を保持する。
      互換維持のため、未設定時は None。
    """

    id: str
    label: str
    source: SourceType
    category: str
    value_type: ValueType
    default_value: Any
    range_hint: RangeHint | None = None
    help_text: str | None = None
    vector_hint: VectorRangeHint | None = None
    supported: bool = True
    choices: list[str] | None = None
    # string 入力用 UI ヒント（明示制御）
    string_multiline: bool = False
    string_height: int | None = None
    # 並び順のヒント（effect グループで使用）
    pipeline_uid: str | None = None
    step_index: int | None = None
    param_order: int | None = None


@dataclass
class ParameterValue:
    """最新の値・上書き状態を保持する。"""

    original: Any
    override: Any | None = None
    timestamp: float = field(default_factory=time.time)

    def resolve(self) -> Any:
        # GUI > original のみ
        if self.override is not None:
            return self.override
        return self.original


@dataclass(frozen=True)
class OverrideResult:
    """override 操作の結果を GUI 側へ返す。"""

    value: Any
    clamped: bool
    source: OverrideSource


Subscriber = Callable[[Iterable[str]], None]


class ParameterStore:
    """パラメータメタデータと値を集中管理する。"""

    def __init__(self) -> None:
        self._descriptors: dict[str, ParameterDescriptor] = {}
        self._values: dict[str, ParameterValue] = {}
        self._listeners: list[Subscriber] = []
        self._lock = RLock()
        # CC 関連（プロバイダとバインディング）
        self._cc_provider: Callable[[], Mapping[int, float]] | None = None
        self._cc_bindings: dict[str, int] = {}

    # --- 登録 / 問合せ ---
    def register(self, descriptor: ParameterDescriptor, value: Any) -> None:
        """Descriptor を登録し、初期値を保存する。"""
        changed: set[str] = set()
        with self._lock:
            if descriptor.id not in self._descriptors:
                self._descriptors[descriptor.id] = descriptor
                changed.add(descriptor.id)
            if descriptor.id not in self._values:
                self._values[descriptor.id] = ParameterValue(original=value)
                changed.add(descriptor.id)
            else:
                current = self._values[descriptor.id]
                if current.original != value:
                    current.original = value
                    current.timestamp = time.time()
                    changed.add(descriptor.id)
        if changed:
            self._notify(changed)

    def descriptors(self) -> list[ParameterDescriptor]:
        with self._lock:
            return list(self._descriptors.values())

    def get_descriptor(self, param_id: str) -> ParameterDescriptor:
        with self._lock:
            return self._descriptors[param_id]

    # --- 値操作 ---
    def resolve(self, param_id: str, original: Any) -> Any:
        """元値を更新しつつ override を適用した値を返す。"""
        with self._lock:
            if param_id not in self._values:
                self._values[param_id] = ParameterValue(original=original)
                value = self._values[param_id]
            else:
                value = self._values[param_id]
                if value.original != original:
                    value.original = original
                    value.timestamp = time.time()
        return value.resolve()

    def current_value(self, param_id: str) -> Any:
        with self._lock:
            entry = self._values.get(param_id)
            if entry is None:
                return None
            return entry.resolve()

    def original_value(self, param_id: str) -> Any:
        with self._lock:
            entry = self._values.get(param_id)
            if entry is None:
                return None
            return entry.original

    def set_override(
        self,
        param_id: str,
        value: Any,
    ) -> OverrideResult:
        # 実値はそのまま保持し、ここではクランプしない
        clamped = False
        with self._lock:
            entry = self._values.setdefault(param_id, ParameterValue(original=value))
            entry.override = value
            entry.timestamp = time.time()
        self._notify({param_id})
        return OverrideResult(value=value, clamped=clamped, source="gui")

    # Reset 機能は削除（要件簡素化）

    # Reset 機能は削除（要件簡素化）

    # --- リスナー ---
    def subscribe(self, listener: Subscriber) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener: Subscriber) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _notify(self, param_ids: Iterable[str]) -> None:
        ids = list(param_ids)
        if not ids:
            return
        for listener in list(self._listeners):
            try:
                listener(ids)
            except Exception:
                continue

    # --- デバッグ/ユーティリティ ---
    def dump_state(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for key, desc in self._descriptors.items():
                value = self._values.get(key)
                result[key] = {
                    "descriptor": desc,
                    "value": value,
                }
            return result

    # --- CC 連携（最小実装） ---
    def set_cc_provider(self, provider: Callable[[], Mapping[int, float]] | None) -> None:
        """現在フレームの CC スナップショットを返す関数を設定する。"""
        self._cc_provider = provider

    def cc_value(self, index: int) -> float:
        """指定 CC 番号の 0..1 値を返す（未定義は 0.0）。"""
        try:
            i = int(index)
        except Exception:
            i = 0
        provider = self._cc_provider
        try:
            mapping = provider() if callable(provider) else {}
            v = float(mapping.get(i, 0.0))
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v
        except Exception:
            return 0.0

    def bind_cc(self, param_id: str, index: int | None) -> None:
        """パラメータに CC 番号をバインド/解除する（None で解除）。"""
        with self._lock:
            if index is None:
                self._cc_bindings.pop(param_id, None)
            else:
                try:
                    i = int(index)
                except Exception:
                    i = 0
                # 範囲は軽くクランプ（0..127 想定）
                if i < 0:
                    i = 0
                if i > 127:
                    i = 127
                self._cc_bindings[param_id] = i

    def cc_binding(self, param_id: str) -> int | None:
        """バインド済み CC 番号を返す（未バインドは None）。"""
        with self._lock:
            return self._cc_bindings.get(param_id)

    def all_cc_bindings(self) -> dict[str, int]:
        """すべての CC バインディングを返す（保存/デバッグ用）。"""
        with self._lock:
            return dict(self._cc_bindings)


@dataclass(frozen=True)
class ParameterLayoutConfig:
    """GUI 表示用レイアウト設定。"""

    row_height: int = 28
    # 後方互換のための単一パディング（未指定時のフォールバック）。
    padding: int = 8
    font_size: int = 12
    value_precision: int = 6
    # ラベル:値（スライダー）列の比率（0.1..0.9）。既定は等分（0.5）。
    label_column_ratio: float = 0.5
    # 残り領域に対する Bars:CC の比率（Bars 側の割合、0.05..0.95）
    bars_cc_ratio: float = 0.7
    # CC 入力ボックスの固定幅（px）
    cc_box_width: int = 24

    # ---- 詳細余白（X, Y）: YAML から配列で受け取り、未指定は padding をフォールバック ----
    # 表全体のセル余白（行間・列間）。Vector/Scalar のバー用内側テーブルにも適用。
    cell_padding_x: int = 8
    cell_padding_y: int = 8
    # 同一グループ内のウィジェット間間隔。CC 入力の横並びに適用。
    item_spacing_x: int = 8
    item_spacing_y: int = 8
    # ウィジェット枠内（InputText/Checkbox/Slider 等）の内側余白。
    frame_padding_x: int = 8
    frame_padding_y: int = 4
    # ウィンドウの内側余白（ビューポートの端からの余白）。
    window_padding_x: int = 8
    window_padding_y: int = 8

    def derive_range(self, *, name: str, value_type: ValueType, default_value: Any) -> RangeHint:
        """最小構成の既定レンジ: 数値は 0..1、bool は 0/1。"""
        if value_type == "bool":
            return RangeHint(0, 1, step=1)
        if value_type == "int":
            return RangeHint(0, 1, step=1)
        # float/enum/vector も 0..1 を既定とする
        return RangeHint(0.0, 1.0)

    def derive_vector_range(self, *, dim: int = 3) -> VectorRangeHint:
        """ベクトルの既定レンジ（各軸 0..1、step 未指定）。"""
        dim = 4 if dim >= 4 else 3
        if dim == 4:
            return VectorRangeHint(
                min_values=(0.0, 0.0, 0.0, 0.0),
                max_values=(1.0, 1.0, 1.0, 1.0),
                steps=(None, None, None, None),
                scale="linear",
            )
        return VectorRangeHint(
            min_values=(0.0, 0.0, 0.0),
            max_values=(1.0, 1.0, 1.0),
            steps=(None, None, None),
            scale="linear",
        )


@dataclass(frozen=True)
class ParameterWindowConfig:
    """Parameter GUI ウィンドウの寸法/タイトル設定。

    - 設定未指定時は `width=420`, `height=640`, `title="Parameters"` を用いる。
    """

    width: int = 420
    height: int = 640
    title: str = "Parameters"


@dataclass(frozen=True)
class ParameterThemeConfig:
    """Parameter GUI のテーマ（スタイル/色）設定。

    - Dear PyGui の StyleVar/ThemeCol に相当するキーを受け取り、そのまま適用する。
    - キーは存在するもののみを適用し、欠損はスキップするフェイルソフト設計。
    - colors は RGBA (0..1) を前提とする。
    """

    style: dict[str, Any] = field(default_factory=dict)
    colors: dict[str, Any] = field(default_factory=dict)
    # カテゴリ別の色設定（shape/pipeline）。任意キー辞書とし、フェイルソフトに適用する。
    # 例:
    #   {
    #       "shape": {"header": [r,g,b,a], "header_hovered": [...], "header_active": [...]},
    #       "pipeline": {"header": [...], "header_hovered": [...], "header_active": [...]}
    #   }
    categories: dict[str, Any] = field(default_factory=dict)


class ParameterRegistry:
    """call ごとの出現回数を追跡するためのユーティリティ。"""

    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)

    def next_index(self, key: str) -> int:
        current = self._counters[key]
        self._counters[key] = current + 1
        return current

    def reset(self) -> None:
        self._counters.clear()
