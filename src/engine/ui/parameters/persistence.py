"""
どこで: `engine.ui.parameters` の永続化ヘルパ。
何を: Parameter GUI の override 値を JSON に保存/復元する（スクリプト単位）。
なぜ: 次回実行時に前回の GUI 調整値を反映し、作業を継続しやすくするため。

仕様（要点）:
- 保存先: 既定 `data/gui/<script_stem>.json`。設定 `parameter_gui.state_dir` で上書き可。
- 保存対象: `ParameterStore` で original と異なる current（override 適用後）のみ。
- 量子化: float は RangeHint/VectorRangeHint の step を優先、未指定は 1e-6
  （環境変数 `PXD_PIPELINE_QUANT_STEP` があればそれを用いる）。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from util.utils import load_config

from .state import ParameterDescriptor, ParameterStore


def _default_step() -> float:
    try:
        from common.settings import get as _get_settings

        return float(_get_settings().PIPELINE_QUANT_STEP)
    except Exception:
        env = os.getenv("PXD_PIPELINE_QUANT_STEP")
        try:
            return float(env) if env is not None else 1e-6
        except Exception:
            return 1e-6


def _quantize_value(value: Any, *, step: float | None) -> Any:
    if step is None:
        return value
    try:
        return round(float(value) / float(step)) * float(step)
    except Exception:
        return value


def _resolve_state_dir() -> Path:
    cfg = load_config() or {}
    pg = cfg.get("parameter_gui", {}) if isinstance(cfg, dict) else {}
    state_dir = None
    if isinstance(pg, dict):
        state_dir = pg.get("state_dir")
    if isinstance(state_dir, str) and state_dir.strip():
        return Path(state_dir)
    return Path.cwd() / "data" / "gui"


def _state_path_for_script(script_path: str | None) -> Path:
    stem = Path(script_path or sys.argv[0]).stem or "script"
    return _resolve_state_dir() / f"{stem}.json"


def _descriptor_step(desc: ParameterDescriptor) -> float | None:
    """Descriptor に付随する推奨 step を返す。

    - vector: `vector_hint.steps` の先頭を優先し、欠損は None。
    - scalar: `range_hint.step` を優先。
    - いずれも無ければ None。
    """
    try:
        if desc.value_type == "vector" and desc.vector_hint is not None:
            steps = desc.vector_hint.steps
            # 最初の有効な数値を採用
            for s in steps:
                if isinstance(s, (int, float)):
                    return float(s)
            return None
        if desc.range_hint is not None and isinstance(desc.range_hint.step, (int, float)):
            return float(desc.range_hint.step)
    except Exception:
        return None
    return None


def _normalize_for_json(value: Any) -> Any:
    """JSON 化できる最小限の変換（tuple→list）。"""
    if isinstance(value, tuple):
        return list(value)
    return value


def save_overrides(store: ParameterStore, *, script_path: str | None = None) -> Path | None:
    """ParameterStore の override を JSON に保存する。

    失敗時は None を返す（フェイルソフト）。
    """
    try:
        path = _state_path_for_script(script_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        overrides: dict[str, Any] = {}
        step_default = _default_step()
        # Descriptor 一覧から original と current の差分のみ保存
        for desc in store.descriptors():
            pid = desc.id
            cur = store.current_value(pid)
            org = store.original_value(pid)
            if cur is None:
                continue
            # 比較は量子化後に行い、微小差分での保存を避ける
            step_hint = _descriptor_step(desc) or step_default

            def _q(x: Any) -> Any:
                if desc.value_type == "vector":
                    try:
                        seq = list(x) if isinstance(x, (list, tuple)) else []
                        if not seq:
                            return []
                        return [
                            _quantize_value(v, step=step_hint) if isinstance(v, (int, float)) else v
                            for v in seq
                        ]
                    except Exception:
                        return x
                if isinstance(x, (int, bool)):
                    return x
                if isinstance(x, float):
                    return _quantize_value(x, step=step_hint)
                return x

            cur_q = _q(cur)
            org_q = _q(org)
            if cur_q == org_q:
                continue

            # 保存値（JSON 化）
            to_save = cur_q
            if isinstance(to_save, tuple):
                to_save = list(to_save)
            overrides[pid] = to_save

        # CC バインディング（セッション永続化: 要件対応）
        try:
            cc_bindings = store.all_cc_bindings()
        except Exception:
            cc_bindings = {}

        # UI レンジ（min/max）オーバーライド
        try:
            ranges_raw = store.all_range_overrides()
        except Exception:
            ranges_raw = {}
        ranges: dict[str, Any] = {}
        if isinstance(ranges_raw, dict):
            for pid, pair in ranges_raw.items():
                try:
                    mn, mx = pair
                    ranges[str(pid)] = {
                        "min": float(mn),
                        "max": float(mx),
                    }
                except Exception:
                    continue

        data = {
            "version": 2,
            "script": str(script_path or sys.argv[0]),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "overrides": overrides,
            "cc_bindings": cc_bindings,
            "ranges": ranges,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def load_overrides(store: ParameterStore, *, script_path: str | None = None) -> int:
    """JSON から override をロードし、ParameterStore に適用する。

    戻り値は適用件数。失敗時は 0。
    """
    try:
        path = _state_path_for_script(script_path)
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        overrides = data.get("overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}

        applied = 0
        # 有効な Descriptor のみ適用（未知キー/型不一致はスキップ）
        index: dict[str, ParameterDescriptor] = {d.id: d for d in store.descriptors()}
        for pid, value in overrides.items():
            desc = index.get(pid)
            if desc is None:
                continue
            try:
                if desc.value_type == "vector":
                    if not isinstance(value, (list, tuple)):
                        continue
                    vec = list(value)
                    # 次元は default_value に合わせて切り詰め/補完しない（そのまま）
                    store.set_override(pid, tuple(float(v) for v in vec))
                    applied += 1
                    continue
                if desc.value_type == "bool":
                    store.set_override(pid, bool(value))
                    applied += 1
                    continue
                if desc.value_type == "int":
                    store.set_override(pid, int(value))
                    applied += 1
                    continue
                if desc.value_type == "float":
                    store.set_override(pid, float(value))
                    applied += 1
                    continue
                if desc.value_type == "enum":
                    # 列挙は文字列として保存/復元
                    store.set_override(pid, str(value))
                    applied += 1
                    continue
                if desc.value_type == "string":
                    store.set_override(pid, str(value))
                    applied += 1
                    continue
                # 非対応型はスキップ
            except Exception:
                continue
        return applied
    except Exception:
        return 0

    finally:
        # CC バインディングと UI レンジオーバーライドの復元（存在時）
        try:
            with open(_state_path_for_script(script_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            # CC
            cc_map = data.get("cc_bindings", {})
            if isinstance(cc_map, dict):
                valid_ids = {d.id for d in store.descriptors()}
                for pid, idx in cc_map.items():
                    accept = False
                    if pid in valid_ids:
                        accept = True
                    else:
                        # ベクトル成分（pid like "{desc.id}::x"）も許可
                        try:
                            base_id = str(pid).split("::", 1)[0]
                            if base_id in valid_ids:
                                accept = True
                        except Exception:
                            accept = False
                    if not accept:
                        continue
                    try:
                        store.bind_cc(str(pid), int(idx))
                    except Exception:
                        continue
            # UI ranges
            ranges = data.get("ranges", {})
            if isinstance(ranges, dict):
                for pid, meta in ranges.items():
                    try:
                        if isinstance(meta, dict):
                            mn = meta.get("min")
                            mx = meta.get("max")
                        elif isinstance(meta, (list, tuple)) and len(meta) >= 2:
                            mn, mx = meta[0], meta[1]
                        else:
                            continue
                        store.set_range_override(str(pid), float(mn), float(mx))
                    except Exception:
                        continue
        except Exception:
            pass


__all__ = ["save_overrides", "load_overrides"]
