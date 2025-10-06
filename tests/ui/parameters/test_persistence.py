import json
from pathlib import Path

import pytest

from engine.ui.parameters.persistence import load_overrides, save_overrides
from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterStore,
    RangeHint,
    VectorRangeHint,
)


def _make_descriptors():
    return [
        ParameterDescriptor(
            id="shape.circle#0.radius",
            label="circle · radius",
            source="shape",
            category="shape",
            value_type="float",
            default_value=1.0,
            range_hint=RangeHint(0.0, 10.0),
        ),
        ParameterDescriptor(
            id="effect.trim#0.count",
            label="trim · count",
            source="effect",
            category="effect",
            value_type="int",
            default_value=1,
            range_hint=RangeHint(0, 10, step=1),
        ),
        ParameterDescriptor(
            id="effect.switch#0.enable",
            label="switch · enable",
            source="effect",
            category="effect",
            value_type="bool",
            default_value=True,
        ),
        ParameterDescriptor(
            id="effect.mode#0.option",
            label="mode · option",
            source="effect",
            category="effect",
            value_type="enum",
            default_value="opt1",
        ),
        ParameterDescriptor(
            id="effect.move#0.offset",
            label="move · offset",
            source="effect",
            category="effect",
            value_type="vector",
            default_value=(0.0, 0.0, 0.0),
            vector_hint=VectorRangeHint(
                min_values=(0.0, 0.0, 0.0),
                max_values=(10.0, 10.0, 10.0),
                steps=(None, None, None),
            ),
        ),
    ]


def _register_defaults(store: ParameterStore, descriptors: list[ParameterDescriptor]) -> None:
    for d in descriptors:
        store.register(d, d.default_value)


def test_persistence_roundtrip_basic_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "util.utils.load_config",
        lambda: {"parameter_gui": {"state_dir": str(tmp_path)}},
        raising=True,
    )

    store = ParameterStore()
    descs = _make_descriptors()
    _register_defaults(store, descs)

    # override を設定
    store.set_override("shape.circle#0.radius", 3.25)
    store.set_override("effect.trim#0.count", 4)
    store.set_override("effect.switch#0.enable", False)
    store.set_override("effect.mode#0.option", "opt2")
    store.set_override("effect.move#0.offset", (1.0, 2.0, 3.0))

    out = save_overrides(store, script_path=str(tmp_path / "demo.py"))
    assert out is not None and out.exists()

    # 新しい Store に復元
    store2 = ParameterStore()
    _register_defaults(store2, descs)
    applied = load_overrides(store2, script_path=str(tmp_path / "demo.py"))
    assert applied >= 5

    assert store2.current_value("shape.circle#0.radius") == pytest.approx(3.25, abs=1e-6)
    assert store2.current_value("effect.trim#0.count") == 4
    assert store2.current_value("effect.switch#0.enable") is False
    assert store2.current_value("effect.mode#0.option") == "opt2"
    assert tuple(store2.current_value("effect.move#0.offset")) == pytest.approx(
        (1.0, 2.0, 3.0), abs=1e-6
    )


def test_persistence_saves_only_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "util.utils.load_config",
        lambda: {"parameter_gui": {"state_dir": str(tmp_path)}},
        raising=True,
    )
    store = ParameterStore()
    desc = ParameterDescriptor(
        id="shape.square#0.size",
        label="square · size",
        source="shape",
        category="shape",
        value_type="float",
        default_value=2.0,
        range_hint=RangeHint(0.0, 10.0),
    )
    store.register(desc, 2.0)

    # override 未設定 → 空の overrides
    p = save_overrides(store, script_path=str(tmp_path / "sq.py"))
    assert p is not None
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data.get("overrides"), dict)
    assert not data["overrides"]

    # override 設定 → overrides に含まれる
    store.set_override(desc.id, 5.5)
    p2 = save_overrides(store, script_path=str(tmp_path / "sq.py"))
    data2 = json.loads(p2.read_text(encoding="utf-8"))
    assert desc.id in data2["overrides"]


def test_persistence_ignores_unknown_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "util.utils.load_config",
        lambda: {"parameter_gui": {"state_dir": str(tmp_path)}},
        raising=True,
    )
    # 既存のファイルに未知キーを混在
    path = tmp_path / "unknown.py"
    store = ParameterStore()
    desc = ParameterDescriptor(
        id="effect.alpha#0.value",
        label="alpha · value",
        source="effect",
        category="effect",
        value_type="float",
        default_value=0.2,
        range_hint=RangeHint(0.0, 1.0),
    )
    store.register(desc, 0.2)
    store.set_override(desc.id, 0.7)
    p = save_overrides(store, script_path=str(path))
    assert p is not None
    data = json.loads(p.read_text(encoding="utf-8"))
    data["overrides"]["unknown.key#0.value"] = 123
    (tmp_path / "data" / "gui").mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")

    store2 = ParameterStore()
    store2.register(desc, 0.2)
    applied = load_overrides(store2, script_path=str(path))
    # 未知キーは無視され、既知キーは適用される
    assert applied >= 1
    assert store2.current_value(desc.id) == pytest.approx(0.7, abs=1e-6)
