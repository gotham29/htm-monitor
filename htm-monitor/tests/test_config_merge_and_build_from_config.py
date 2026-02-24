# tests/test_config_merge_and_build_from_config.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


def _write_yaml(p: Path, blob: Dict[str, Any]) -> str:
    p.write_text(yaml.safe_dump(blob, sort_keys=False))
    return str(p)


class _StubFeature:
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params


class _StubHTMmodel:
    """
    Minimal stub to let build_from_config() construct models without importing htm.bindings.
    Engine tests already cover merge logic; here we only care build_from_config wiring.
    """
    def __init__(self, features: Dict[str, Any], models_params: dict, return_pred_count: bool, name: str):
        self.features = features
        self.models_params = models_params
        self.return_pred_count = return_pred_count
        self.name = name


class _StubEngine:
    def __init__(self, models, model_sources, on_missing: str = "skip"):
        self.models = models
        self.model_sources = model_sources
        self.on_missing = on_missing


class _StubDecision:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.threshold = kwargs.get("threshold")


def test_merge_keeps_default_nested_params_when_user_overrides_other_sections(tmp_path: Path, monkeypatch):
    """
    This test encodes the *desired* behavior:
    - user config overrides e.g. decision.threshold
    - but does NOT blow away defaults' nested htm_params.tm subtree

    If merge is shallow, this will fail (good: it forces a conscious decision).
    """
    # Import inside test so monkeypatch applies cleanly
    import htm_monitor.utils.config as cfgmod

    # monkeypatch heavy deps
    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {
            "tm": {
                "cellsPerColumn": 32,
                "activationThreshold": 13,
                "minThreshold": 10,
                "newSynapseCount": 20,
            },
            "anomaly_likelihood": {"learningPeriod": 150, "estimationSamples": 50},
        },
        "features": {
            "timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"},
            "x": {"type": "float", "size": 2048, "activeBits": 40, "numBuckets": 130, "minVal": 0, "maxVal": 1, "seed": 1},
        },
        "models": {"x_model": {"source": "sx", "features": ["timestamp", "x"]}},
        "data": {"timebase": {"on_missing": "skip"}, "sources": []},
        "decision": {"threshold": 0.99, "method": "max", "score_key": "likelihood"},
        "plot": {"enable": False},
    }

    user = {
        # user overrides decision only
        "decision": {"threshold": 0.5, "method": "max", "score_key": "likelihood"},
    }

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, user)

    out = cfgmod.build_from_config(str(p_def), str(p_usr))

    # build_from_config might return (cfg, engine, decision) OR (cfg, engine, decision, model_sources).
    # Normalize for the test:
    assert isinstance(out, tuple)
    cfg = out[0]

    assert cfg["decision"]["threshold"] == 0.5

    # The crux: default nested htm_params must remain present.
    assert "htm_params" in cfg
    assert "tm" in cfg["htm_params"]
    assert cfg["htm_params"]["tm"]["cellsPerColumn"] == 32
    assert cfg["htm_params"]["tm"]["activationThreshold"] == 13


def test_build_from_config_returns_model_sources_and_normalizes_source_forms(tmp_path: Path, monkeypatch):
    """
    Locks the new run_pipeline expectation:
      cfg, engine, decision, model_sources = build_from_config(...)
    And model_sources must be dict[str, list[str]].
    """
    import htm_monitor.utils.config as cfgmod

    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {"tm": {"cellsPerColumn": 32, "activationThreshold": 13, "minThreshold": 10, "newSynapseCount": 20}},
        "features": {
            "timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"},
            "a": {"type": "float", "size": 10, "activeBits": 2, "numBuckets": 5, "minVal": 0, "maxVal": 1, "seed": 1},
            "b": {"type": "float", "size": 10, "activeBits": 2, "numBuckets": 5, "minVal": 0, "maxVal": 1, "seed": 2},
        },
        "models": {
            "a_model": {"source": "sa", "features": ["timestamp", "a"]},
            "b_model": {"sources": ["sb1", "sb2"], "features": ["timestamp", "b"]},
        },
        "data": {"timebase": {"on_missing": "hold_last"}, "sources": []},
        "decision": {"threshold": 0.9, "method": "kofn_window", "k": 1, "window": {"size": 3, "per_model_hits": 1}, "score_key": "likelihood"},
    }
    user = {}

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, user)

    out = cfgmod.build_from_config(str(p_def), str(p_usr))

    assert isinstance(out, tuple)
    assert len(out) == 4, "build_from_config must return (cfg, engine, decision, model_sources)"
    cfg, engine, decision, model_sources = out

    assert isinstance(model_sources, dict)
    assert model_sources["a_model"] == ["sa"]
    assert model_sources["b_model"] == ["sb1", "sb2"]

    # Ensure Engine received normalized sources (and on_missing)
    assert engine.on_missing == "hold_last"
    assert engine.model_sources["a_model"] == ["sa"]
    assert engine.model_sources["b_model"] == ["sb1", "sb2"]


def test_rejects_unknown_top_level_keys(tmp_path: Path, monkeypatch):
    import htm_monitor.utils.config as cfgmod

    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {"tm": {"cellsPerColumn": 32, "activationThreshold": 13, "minThreshold": 10, "newSynapseCount": 20}},
        "features": {"timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"}, "x": {"type": "float"}},
        "models": {"x_model": {"source": "sx", "features": ["timestamp", "x"]}},
        "data": {"timebase": {"on_missing": "skip"}, "sources": []},
        "decision": {"threshold": 0.9, "method": "max", "score_key": "likelihood"},
    }
    user = {"typo_section": {"oops": True}}

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, user)

    with pytest.raises(ValueError, match="unknown top-level"):
        cfgmod.build_from_config(str(p_def), str(p_usr))


def test_requires_timestamp_feature(tmp_path: Path, monkeypatch):
    import htm_monitor.utils.config as cfgmod

    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {"tm": {"cellsPerColumn": 32, "activationThreshold": 13, "minThreshold": 10, "newSynapseCount": 20}},
        "features": {"x": {"type": "float"}},  # missing timestamp
        "models": {"x_model": {"source": "sx", "features": ["x"]}},
        "data": {"timebase": {"on_missing": "skip"}, "sources": []},
        "decision": {"threshold": 0.9, "method": "max", "score_key": "likelihood"},
    }

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, {})

    with pytest.raises(ValueError, match="must include a 'timestamp'"):
        cfgmod.build_from_config(str(p_def), str(p_usr))


def test_model_rejects_unknown_feature_reference(tmp_path: Path, monkeypatch):
    import htm_monitor.utils.config as cfgmod

    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {"tm": {"cellsPerColumn": 32, "activationThreshold": 13, "minThreshold": 10, "newSynapseCount": 20}},
        "features": {"timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"}, "x": {"type": "float"}},
        "models": {"bad_model": {"source": "sx", "features": ["timestamp", "DOES_NOT_EXIST"]}},
        "data": {"timebase": {"on_missing": "skip"}, "sources": []},
        "decision": {"threshold": 0.9, "method": "max", "score_key": "likelihood"},
    }

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, {})

    with pytest.raises(ValueError, match="references unknown feature"):
        cfgmod.build_from_config(str(p_def), str(p_usr))


def test_decision_window_must_be_mapping_if_provided(tmp_path: Path, monkeypatch):
    import htm_monitor.utils.config as cfgmod

    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {"tm": {"cellsPerColumn": 32, "activationThreshold": 13, "minThreshold": 10, "newSynapseCount": 20}},
        "features": {"timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"}, "x": {"type": "float"}},
        "models": {"x_model": {"source": "sx", "features": ["timestamp", "x"]}},
        "data": {"timebase": {"on_missing": "skip"}, "sources": []},
        "decision": {
            "threshold": 0.9,
            "method": "kofn_window",
            "k": 1,
            "window": "not-a-mapping",
            "score_key": "likelihood",
        },
    }

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, {})

    with pytest.raises(ValueError, match="decision.window must be a mapping"):
        cfgmod.build_from_config(str(p_def), str(p_usr))


def test_on_missing_is_lowercased_before_engine(tmp_path: Path, monkeypatch):
    import htm_monitor.utils.config as cfgmod

    monkeypatch.setattr(cfgmod, "Feature", _StubFeature)
    monkeypatch.setattr(cfgmod, "HTMmodel", _StubHTMmodel)
    monkeypatch.setattr(cfgmod, "Engine", _StubEngine)
    monkeypatch.setattr(cfgmod, "Decision", _StubDecision)

    defaults = {
        "htm_params": {"tm": {"cellsPerColumn": 32, "activationThreshold": 13, "minThreshold": 10, "newSynapseCount": 20}},
        "features": {"timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"}, "x": {"type": "float"}},
        "models": {"x_model": {"source": "sx", "features": ["timestamp", "x"]}},
        "data": {"timebase": {"on_missing": "HOLD_LAST"}, "sources": []},
        "decision": {"threshold": 0.9, "method": "max", "score_key": "likelihood"},
    }

    p_def = tmp_path / "defaults.yaml"
    p_usr = tmp_path / "user.yaml"
    _write_yaml(p_def, defaults)
    _write_yaml(p_usr, {})

    cfg, engine, decision, model_sources = cfgmod.build_from_config(str(p_def), str(p_usr))
    assert engine.on_missing == "hold_last"
