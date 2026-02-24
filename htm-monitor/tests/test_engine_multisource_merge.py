# tests/test_engine_multisource_merge.py

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pytest

from htm_monitor.orchestration.engine import Engine


@dataclass
class _StubModel:
    """
    Minimal stand-in for HTMmodel.
    We only care that Engine passes the right merged features_data.
    """
    required_keys: Sequence[str]
    calls: List[Dict[str, Any]]

    def run(self, features_data: Mapping[str, Any], timestep: int, learn: bool):
        # validate required keys exist and are not None
        for k in self.required_keys:
            assert k in features_data, f"missing key {k}"
            assert features_data[k] is not None, f"{k} is None"

        self.calls.append({"features_data": dict(features_data), "timestep": timestep, "learn": learn})

        # raw, likelihood, pcount
        return 0.1, 0.2, None


def test_engine_merges_across_multiple_sources_happy_path():
    model = _StubModel(required_keys=["timestamp", "a", "b", "c"], calls=[])
    models = {"trio_model": model}

    # Engine implementation may store model_sources as str or list; pass list (multi-source case)
    model_sources = {"trio_model": ["s1", "s2", "s3"]}
    eng = Engine(models=models, model_sources=model_sources, on_missing="skip")

    rows_by_source = {
        "s1": {"timestamp": "t0", "a": 1.0},
        "s2": {"timestamp": "t0", "b": 2.0},
        "s3": {"timestamp": "t0", "c": 3.0},
    }

    out = eng.step(rows_by_source, timestep=0)
    assert "trio_model" in out
    assert len(model.calls) == 1
    merged = model.calls[0]["features_data"]
    assert merged["timestamp"] == "t0"
    assert merged["a"] == 1.0 and merged["b"] == 2.0 and merged["c"] == 3.0


def test_engine_skip_when_required_feature_missing_under_union_timebase():
    model = _StubModel(required_keys=["timestamp", "a", "b"], calls=[])
    models = {"m": model}
    eng = Engine(models=models, model_sources={"m": ["s1", "s2"]}, on_missing="skip")

    # only s1 present at this timestep => missing b
    rows_by_source = {"s1": {"timestamp": "t0", "a": 1.0}}
    out = eng.step(rows_by_source, timestep=0)

    assert out == {} or "m" not in out
    assert len(model.calls) == 0


def test_engine_handles_none_as_missing_and_can_take_value_from_other_source():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    models = {"m": model}
    eng = Engine(models=models, model_sources={"m": ["s1", "s2"]}, on_missing="skip")

    rows_by_source = {
        "s1": {"timestamp": "t0", "a": None},
        "s2": {"timestamp": "t0", "a": 123.0},
    }
    out = eng.step(rows_by_source, timestep=0)

    assert "m" in out
    assert len(model.calls) == 1
    assert model.calls[0]["features_data"]["a"] == 123.0


def test_engine_rejects_unknown_on_missing_mode():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="nope")

    try:
        eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
        assert False, "Expected ValueError for unsupported on_missing mode"
    except ValueError:
        pass


def test_engine_rejects_timestamp_mismatch_across_sources():
    model = _StubModel(required_keys=["timestamp", "a", "b"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1", "s2"]}, on_missing="skip")

    rows_by_source = {
        "s1": {"timestamp": "t0", "a": 1.0},
        "s2": {"timestamp": "t1", "b": 2.0},  # mismatch
    }

    try:
        eng.step(rows_by_source, timestep=0)
        assert False, "Expected ValueError for timestamp mismatch"
    except ValueError:
        pass


def test_engine_hold_last_reuses_previous_values_but_keeps_current_timestamp():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # t0: good row -> establish last_good
    out0 = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    assert "m" in out0
    assert len(model.calls) == 1
    assert model.calls[0]["features_data"] == {"timestamp": "t0", "a": 1.0}

    # t1: missing a -> should still run using last_good 'a', but timestamp should be current
    out1 = eng.step({"s1": {"timestamp": "t1", "a": None}}, timestep=1)
    assert "m" in out1
    assert len(model.calls) == 2
    assert model.calls[1]["features_data"] == {"timestamp": "t1", "a": 1.0}


def test_engine_hold_last_raises_if_timestamp_missing_everywhere():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # establish last_good
    out0 = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    assert "m" in out0

    # next step: timestamp missing entirely (not even present as key)
    with pytest.raises(ValueError, match="timestamp missing"):
        eng.step({"s1": {"a": None}}, timestep=1)


def test_engine_raises_if_model_has_no_sources_configured():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={}, on_missing="skip")

    try:
        eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
        assert False, "Expected ValueError when model has no sources configured"
    except ValueError:
        pass


def test_engine_feature_precedence_respects_source_order_first_non_none_wins():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1", "s2"]}, on_missing="skip")

    rows_by_source = {
        "s1": {"timestamp": "t0", "a": 111.0},
        "s2": {"timestamp": "t0", "a": 222.0},
    }
    out = eng.step(rows_by_source, timestep=0)
    assert "m" in out
    assert len(model.calls) == 1
    assert model.calls[0]["features_data"]["a"] == 111.0


def test_engine_union_hold_last_allows_model_missing_row_if_system_timestamp_exists():
    """
    Under union timebase, rows_by_source at a timestep may include only some sources.
    With on_missing=hold_last, a model whose source is absent should still run using last_good,
    but MUST advance timestamp to the current system timestamp from other sources.
    """
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # t0: establish last_good from s1
    out0 = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    assert "m" in out0
    assert model.calls[0]["features_data"] == {"timestamp": "t0", "a": 1.0}

    # t1: s1 missing entirely, but system timestamp exists via s2
    out1 = eng.step({"s2": {"timestamp": "t1", "b": 999.0}}, timestep=1)
    assert "m" in out1
    assert model.calls[1]["features_data"] == {"timestamp": "t1", "a": 1.0}


def test_engine_hold_last_rejects_when_timestamp_missing_everywhere():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # Establish last_good at t0
    out0 = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    assert "m" in out0

    # Now timestamp is missing entirely (strict: cannot proceed)
    try:
        eng.step({"s1": {"a": 2.0}}, timestep=1)
        assert False, "Expected ValueError when timestamp missing under hold_last"
    except ValueError:
        pass


def test_engine_output_schema_has_expected_keys():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="skip")
    out = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    m = out["m"]
    for k in ["raw", "likelihood", "p", "anomaly_probability", "log_likelihood", "pcount"]:
        assert k in m
    assert isinstance(m["raw"], float)
    assert isinstance(m["likelihood"], float)


def test_engine_hold_last_uses_global_timestamp_when_model_sources_missing():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    # model reads from s1, but at t1 only s2 arrives (union timebase behavior)
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # t0: establish last_good from s1
    out0 = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    assert "m" in out0
    assert model.calls[-1]["features_data"] == {"timestamp": "t0", "a": 1.0}

    # t1: s1 missing entirely, but some other source supplies the timestep timestamp
    out1 = eng.step({"s2": {"timestamp": "t1", "zzz": 999}}, timestep=1)
    assert "m" in out1
    assert model.calls[-1]["features_data"] == {"timestamp": "t1", "a": 1.0}


def test_engine_hold_last_raises_when_timestamp_missing_everywhere():
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # establish last_good
    eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)

    # next timestep: no timestamps anywhere -> strict failure
    try:
        eng.step({"s2": {"not_timestamp": "x"}}, timestep=1)
        assert False, "Expected ValueError when timestamp missing everywhere"
    except ValueError:
        pass


def test_engine_hold_last_skips_model_when_model_sources_missing_entirely_but_system_has_timestamp():
    """
    Union timebase: timestep emits only the sources that have a row at that time.
    If a model's sources are absent in this timestep, Engine should NOT raise,
    even in hold_last mode, as long as the timestep has a well-defined system timestamp.
    It should reuse last_good for that model (including timestamp overwrite to current)
    only if it can infer the current timestep timestamp from other sources.
    """
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # t0: establish last_good
    out0 = eng.step({"s1": {"timestamp": "t0", "a": 1.0}}, timestep=0)
    assert "m" in out0
    assert model.calls[-1]["features_data"] == {"timestamp": "t0", "a": 1.0}

    # t1: model source s1 missing entirely, but another source carries the system timestamp
    out1 = eng.step({"other": {"timestamp": "t1", "x": 999.0}}, timestep=1)

    # Should run m using last_good values but current system timestamp
    assert "m" in out1
    assert model.calls[-1]["features_data"] == {"timestamp": "t1", "a": 1.0}


def test_engine_raises_if_multiple_timestamps_appear_in_one_union_timestep():
    """
    If the union stream ever emits a timestep containing multiple different timestamps,
    that's an invariant violation and must raise (air-tight).
    """
    model = _StubModel(required_keys=["timestamp", "a"], calls=[])
    eng = Engine(models={"m": model}, model_sources={"m": ["s1"]}, on_missing="hold_last")

    # even if m isn't involved, the timestep itself is inconsistent
    try:
        eng.step(
            {
                "s1": {"timestamp": "t0", "a": 1.0},
                "s2": {"timestamp": "t1", "b": 2.0},
            },
            timestep=0,
        )
        assert False, "Expected ValueError for multiple timestamps in same timestep"
    except ValueError:
        pass