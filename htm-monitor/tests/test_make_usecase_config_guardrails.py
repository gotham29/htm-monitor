#tests/test_make_usecase_config_guardrails.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demo.make_usecase_config import SourceSpec, build_usecase_config


def _write_csv(tmp_path: Path, name: str, cols: dict) -> str:
    p = tmp_path / name
    df = pd.DataFrame(cols)
    df.to_csv(p, index=False)
    return str(p)


def test_sourcespec_rejects_duplicate_columns_in_fields(tmp_path: Path):
    csv = _write_csv(
        tmp_path,
        "a.csv",
        {
            "timestamp": ["2011-01-01 00:00:00", "2011-01-01 00:00:01"],
            "value": [1.0, 2.0],
        },
    )
    with pytest.raises(ValueError, match="Duplicate csv column"):
        SourceSpec(
            name="s",
            path=csv,
            timestamp_col="timestamp",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            fields={"feat1": "value", "feat2": "value"},
        )


def test_build_usecase_config_rejects_missing_timestamp_col(tmp_path: Path):
    csv = _write_csv(
        tmp_path,
        "a.csv",
        {
            "ts": ["2011-01-01 00:00:00", "2011-01-01 00:00:01"],
            "value": [1.0, 2.0],
        },
    )
    src = SourceSpec(
        name="s",
        path=csv,
        timestamp_col="timestamp",  # wrong
        timestamp_format="%Y-%m-%d %H:%M:%S",
        fields={"feat": "value"},
    )
    with pytest.raises(ValueError, match="timestamp_col .* not found"):
        build_usecase_config("uc", [src])


def test_build_usecase_config_rejects_missing_value_col(tmp_path: Path):
    csv = _write_csv(
        tmp_path,
        "a.csv",
        {
            "timestamp": ["2011-01-01 00:00:00", "2011-01-01 00:00:01"],
            "v": [1.0, 2.0],
        },
    )
    src = SourceSpec(
        name="s",
        path=csv,
        timestamp_col="timestamp",
        timestamp_format="%Y-%m-%d %H:%M:%S",
        fields={"feat": "value"},  # wrong
    )
    with pytest.raises(ValueError, match="Selected column\\(s\\) .* not found"):
        build_usecase_config("uc", [src])


def test_build_usecase_config_rejects_rdse_active_bits_gt_size(tmp_path: Path):
    csv = _write_csv(
        tmp_path,
        "a.csv",
        {
            "timestamp": ["2011-01-01 00:00:00", "2011-01-01 00:00:01"],
            "value": [1.0, 2.0],
        },
    )
    src = SourceSpec(
        name="s",
        path=csv,
        timestamp_col="timestamp",
        timestamp_format="%Y-%m-%d %H:%M:%S",
        fields={"feat": "value"},
    )
    with pytest.raises(ValueError, match="rdse_active_bits must be <= rdse_size"):
        build_usecase_config("uc", [src], rdse_size=10, rdse_active_bits=11)