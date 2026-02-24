# tests/test_make_usecase_config.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from demo.make_usecase_config import SourceSpec, build_usecase_config


def _write_csv(tmp_path: Path, name: str, values):
    p = tmp_path / name
    df = pd.DataFrame(
        {
            "timestamp": [f"2011-01-01 00:00:{i:02d}" for i in range(len(values))],
            "value": list(values),
        }
    )
    df.to_csv(p, index=False)
    return str(p)


def test_build_usecase_config_includes_features_models_and_sources(tmp_path: Path):
    csv_a = _write_csv(tmp_path, "a.csv", [0.0, 1.0, 2.0, 3.0, 100.0])  # outlier
    csv_b = _write_csv(tmp_path, "b.csv", [10.0, 11.0, 12.0, 13.0, 14.0])

    sources = [
        SourceSpec(
            name="sa",
            path=csv_a,
            timestamp_col="timestamp",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            fields={"feat_a": "value"},
        ),
        SourceSpec(
            name="sb",
            path=csv_b,
            timestamp_col="timestamp",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            fields={"feat_b": "value"},
        ),
    ]

    cfg = build_usecase_config(
        "my_usecase",
        sources,
        low_q=0.01,
        high_q=0.99,
        margin=0.03,
        seed_base=42,
    )

    # basic top-level sections
    assert "features" in cfg
    assert "models" in cfg
    assert "data" in cfg and "sources" in cfg["data"]
    assert "decision" in cfg
    assert "plot" in cfg

    # timestamp feature exists
    assert "timestamp" in cfg["features"]
    assert cfg["features"]["timestamp"]["type"] == "datetime"

    # features exist and have minVal/maxVal
    for f in ("feat_a", "feat_b"):
        assert f in cfg["features"]
        assert cfg["features"][f]["type"] == "float"
        assert "minVal" in cfg["features"][f]
        assert "maxVal" in cfg["features"][f]
        assert cfg["features"][f]["minVal"] < cfg["features"][f]["maxVal"]

    # model per feature
    assert "feat_a_model" in cfg["models"]
    assert cfg["models"]["feat_a_model"]["features"] == ["timestamp", "feat_a"]
    assert cfg["models"]["feat_a_model"]["sources"] == ["sa"]

    # data sources map canonical feature -> column
    ds = {d["name"]: d for d in cfg["data"]["sources"]}
    assert ds["sa"]["fields"]["feat_a"] == "value"
    assert ds["sb"]["fields"]["feat_b"] == "value"

    # decision uses anomaly_probability by default
    assert cfg["decision"]["score_key"] == "anomaly_probability"