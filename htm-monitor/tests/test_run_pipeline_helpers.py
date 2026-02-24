# tests/test_run_pipeline_helpers.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional

import pytest

from demo.run_pipeline import (
    _parse_gt_timestamps,
    _timebase_union,
    _timebase_intersection,
    run,
)


def test_parse_gt_timestamps_happy_path():
    labels = {"timestamps": ["2011-07-14 10:15:01", "2011-07-20 10:15:01"]}
    out = _parse_gt_timestamps(labels, "%Y-%m-%d %H:%M:%S")
    assert out == {"2011-07-14 10:15:01", "2011-07-20 10:15:01"}


def test_parse_gt_timestamps_rejects_non_list():
    labels = {"timestamps": "2011-07-14 10:15:01"}  # wrong type
    with pytest.raises(ValueError, match="labels.timestamps must be a list"):
        _parse_gt_timestamps(labels, "%Y-%m-%d %H:%M:%S")


def test_parse_gt_timestamps_validates_format():
    labels = {"timestamps": ["NOT_A_TS"]}
    with pytest.raises(ValueError):
        _parse_gt_timestamps(labels, "%Y-%m-%d %H:%M:%S")


def test_timebase_union_emits_min_timestamp_rows():
    # iters: name -> iterator of (dt, row)
    iters = {
        "a": iter([
            (datetime(2020, 1, 1, 0, 0, 0), {"timestamp": "t0", "x": 1.0}),
            (datetime(2020, 1, 1, 0, 0, 2), {"timestamp": "t2", "x": 2.0}),
        ]),
        "b": iter([
            (datetime(2020, 1, 1, 0, 0, 1), {"timestamp": "t1", "y": 10.0}),
            (datetime(2020, 1, 1, 0, 0, 2), {"timestamp": "t2", "y": 11.0}),
        ]),
    }

    out = list(_timebase_union(iters))
    assert out[0] == {"a": {"timestamp": "t0", "x": 1.0}}
    assert out[1] == {"b": {"timestamp": "t1", "y": 10.0}}
    # at t2 both present
    assert out[2] == {
        "a": {"timestamp": "t2", "x": 2.0},
        "b": {"timestamp": "t2", "y": 11.0},
    }


def test_timebase_intersection_only_when_all_equal():
    iters = {
        "a": iter([
            (datetime(2020, 1, 1, 0, 0, 0), {"timestamp": "t0", "x": 1.0}),
            (datetime(2020, 1, 1, 0, 0, 2), {"timestamp": "t2", "x": 2.0}),
        ]),
        "b": iter([
            (datetime(2020, 1, 1, 0, 0, 1), {"timestamp": "t1", "y": 10.0}),
            (datetime(2020, 1, 1, 0, 0, 2), {"timestamp": "t2", "y": 11.0}),
        ]),
    }

    out = list(_timebase_intersection(iters))
    assert out == [{
        "a": {"timestamp": "t2", "x": 2.0},
        "b": {"timestamp": "t2", "y": 11.0},
    }]


class _EngineStub:
    def __init__(self):
        self.calls = 0

    def step(self, row: Mapping[str, Any], timestep: int) -> Dict[str, Dict[str, Any]]:
        self.calls += 1
        # mimic engine output shape: model -> metrics dict
        return {"m1": {"raw": 0.1 * timestep, "likelihood": 0.9}}


class _DecisionStub:
    def __init__(self):
        self.calls = 0

    def step(self, model_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.calls += 1
        return {"system_score": 0.5, "alert": False}


def test_run_yields_and_calls_on_update():
    stream = [{"timestamp": "t0"}, {"timestamp": "t1"}, {"timestamp": "t2"}]
    engine = _EngineStub()
    decision = _DecisionStub()

    seen = []

    def on_update(t: int, row: Mapping[str, Any], model_outputs: Any, result: Any) -> None:
        seen.append((t, row.get("timestamp"), bool(model_outputs), result.get("system_score")))

    results = list(run(stream, engine, decision, on_update=on_update))
    assert len(results) == 3
    assert engine.calls == 3
    assert decision.calls == 3
    assert seen == [
        (0, "t0", True, 0.5),
        (1, "t1", True, 0.5),
        (2, "t2", True, 0.5),
    ]